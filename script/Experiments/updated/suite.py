"""
The pipeline of Large_Test.ipynb as a command-line suite, for comparisons with dhov_rust.

It reads the same VNN-COMP style instances.csv (net,spec,timeout) as `dhov suite`, runs one variant of the old
pipeline on every instance and writes one JSON line per instance with the fields of dhov_rust's results.jsonl
(net, spec, config, verdict, bound, o_value, precheck, layers, times), so compare_results.py can join both.

Variants (all with LP bound tightening of every layer after the first, as the notebook):
* snr       every ReLU layer SNR                                   (dhov_rust configs/snr/*)
* snr_milp  every ReLU layer SNR, the last one exact (MILP)        (configs/snr_milp/*, notebook cell "SNR with MILP")
* dhov      every ReLU layer DHOV, the last one exact (MILP)       (configs/dhov/*, notebook cell "DHOV Verification")

Differences to the notebook, on purpose (see the comparison section in dhov_rust's docs/benchmarks.md):
* The network is taken from the ONNX file as stored (input normalisation folded into the first layer); a final
  ReLU is kept by appending an identity layer and skip_last_layer=True, as the notebook did for the 9x200 net.
* Intermediate bounds are read from the inputs of auto_LiRPA's ReLU nodes (not by node name) and checked against
  sampled points; a final ReLU gets its identity layer's input bounds (the ReLU of the logits' bounds).
* The label comes from the VNNLIB file, not from the MNIST training set at the same index.
* Verified means: the dual bound of the property MILP, widened by the solver tolerance, is < 0. Falsified means:
  a concrete input in the box, checked by a forward pass. The notebook read the incumbent and set BestObjStop =
  1e-3 (add_output_constraints' default), so its "max distance" was the first point found above 1e-3, neither
  the optimum nor a bound.
* Early stop and pre-check as dhov_rust's shipped configs (`[property] stop = "decided"`, `precheck = true`): the
  property MILP stops at the first verifying dual bound or the first relaxation point above 0 (o-values: only the
  sign counts), and the bound method's bound of every o_j - o_c is tried first (auto_LiRPA with a spec matrix).
  --stop optimal / --no-precheck give exact, comparable o-values instead.
* DHOV settings follow dhov_rust's configs/dhov/abstract_dhov.toml (Adam, learning rate 1e-2) unless overridden;
  --optimizer SdLBFGS --learning-rate 1 gives the notebook's optimizer.

Example (from the repository root; the .venv has the pinned packages, see icnn-crown-instructions):
  .venv/bin/python script/Experiments/updated/suite.py --variant snr_milp --bounds alpha-crown \\
      --instances ../dhov_rust/benchmarks/specs/mnist-net_256x4_eps0.026/instances.csv \\
      --nets ../dhov_rust/benchmarks/nets --out results/python/snr_milp_alpha_crown.jsonl
"""
import argparse
import contextlib
import csv
import json
import math
import os
import random
import re
import sys
import time
import traceback

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import gurobipy as grp  # noqa: E402
import onnx  # noqa: E402
import torch  # noqa: E402
from onnx import numpy_helper  # noqa: E402

import script.DHOV.MultiDHOV as multidhov  # noqa: E402
import script.Verification.VerificationBasics as verbas  # noqa: E402
from script.DHOV.Sampling.PerGroupLineSearchSampling import PerGroupLineSearchSamplingStrategy  # noqa: E402
from script.NeuralNets.ICNNFactory import ICNNFactory  # noqa: E402
from script.NeuralNets.Networks import SequentialNN  # noqa: E402
from script.settings import (data_type, device, get_gurobi_params, solver_tolerance_absolute,  # noqa: E402
                             solver_tolerance_relative)

SHAPE_OPS = {"Flatten", "Reshape", "Squeeze", "Unsqueeze", "Identity"}


@contextlib.contextmanager
def quiet():
    """Suppresses the prints of the pipeline and of auto_LiRPA."""
    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
        yield


# ---------------------------------------------------------------------------------------------------------------
# network

def onnx_layers(onnx_path):
    """
    The dense layers of an ONNX file as [(W [out, in], b [out], relu)], in float64, with a leading Sub/Div input
    normalisation folded into the first layer. Raises ValueError for any other structure.
    """
    model = onnx.load(onnx_path)
    graph = model.graph
    constants = {init.name: numpy_helper.to_array(init).astype(np.float64) for init in graph.initializer}
    layers = []
    shift, scale = None, None
    pending_matmul = None
    for node in graph.node:
        op = node.op_type
        if op == "Constant":
            constants[node.output[0]] = numpy_helper.to_array(node.attribute[0].t).astype(np.float64)
        elif op in SHAPE_OPS:
            continue
        elif op in ("Sub", "Div") and not layers:
            value = constants[node.input[1]].reshape(-1)
            if op == "Sub":
                shift = value
            else:
                scale = value
        elif op == "Gemm":
            attrs = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
            w = constants[node.input[1]]
            w = w if attrs.get("transB", 0) else w.T
            if attrs.get("transA", 0):
                raise ValueError("Gemm with transA is not supported")
            b = constants[node.input[2]].reshape(-1) if len(node.input) > 2 else np.zeros(w.shape[0])
            layers.append([attrs.get("alpha", 1.0) * w, attrs.get("beta", 1.0) * b, False])
        elif op == "MatMul":
            pending_matmul = constants[node.input[1]].T
        elif op == "Add" and pending_matmul is not None:
            bias_name = node.input[1] if node.input[1] in constants else node.input[0]
            layers.append([pending_matmul, constants[bias_name].reshape(-1), False])
            pending_matmul = None
        elif op == "Relu":
            if not layers or layers[-1][2]:
                raise ValueError("Relu without a preceding dense layer")
            layers[-1][2] = True
        else:
            raise ValueError("unsupported ONNX op {}".format(op))
    if pending_matmul is not None:
        layers.append([pending_matmul, np.zeros(pending_matmul.shape[0]), False])
    if not layers:
        raise ValueError("no dense layer in {}".format(onnx_path))
    if any(not relu for _, _, relu in layers[:-1]):
        raise ValueError("every layer but the last needs a ReLU")

    # fold (x - shift) / scale into the first layer
    w, b, relu = layers[0]
    n_in = w.shape[1]
    shift = np.zeros(n_in) if shift is None else np.broadcast_to(shift, (n_in,))
    scale = np.ones(n_in) if scale is None else np.broadcast_to(scale, (n_in,))
    w = w / scale
    layers[0] = [w, b - w @ shift, relu]
    return [(w, b, relu) for w, b, relu in layers]


def build_network(onnx_path):
    """
    (SequentialNN, relu_on_output): the ONNX network as the old pipeline needs it. SequentialNN has no ReLU after
    its last layer, so a final ReLU of the ONNX file is kept by appending an identity layer (relu_on_output=True,
    verify with skip_last_layer=True), exactly as Large_Test.ipynb did for mnist_relu_9_200.onnx.
    """
    layers = onnx_layers(onnx_path)
    relu_on_output = layers[-1][2]
    if relu_on_output:
        out = layers[-1][0].shape[0]
        layers.append((np.eye(out), np.zeros(out), False))
    widths = [layers[0][0].shape[1]] + [w.shape[0] for w, _, _ in layers]
    nn = SequentialNN(widths)
    params = list(nn.parameters())
    for k, (w, b, _) in enumerate(layers):
        params[2 * k].data = torch.tensor(w, dtype=data_type).to(device)
        params[2 * k + 1].data = torch.tensor(b, dtype=data_type).to(device)
    for p in params:
        p.requires_grad_(False)
    return nn, relu_on_output


def check_against_onnxruntime(onnx_path, nn, relu_on_output, inputs):
    """Raises if the SequentialNN differs from onnxruntime on inputs (rows of [n, d_in])."""
    import onnxruntime
    session = onnxruntime.InferenceSession(onnx_path)
    meta = session.get_inputs()[0]
    shape = [d if isinstance(d, int) and d > 0 else 1 for d in meta.shape]
    for x in inputs:
        expected = session.run(None, {meta.name: x.reshape(shape).astype(np.float32)})[0].reshape(-1)
        actual = nn(torch.tensor(x, dtype=data_type).unsqueeze(0)).detach().cpu().numpy().reshape(-1)
        if relu_on_output:
            actual = np.maximum(actual, 0)  # the identity layer passes the ReLU output on
        if not np.allclose(actual, expected, rtol=1e-4, atol=1e-4):
            raise RuntimeError("network of {} differs from onnxruntime: {} vs {}".format(onnx_path, actual, expected))


# ---------------------------------------------------------------------------------------------------------------
# specification

def read_spec(vnnlib_path, n_in, n_out):
    """(lower, upper, label): the input box as data_type tensors and the label of the robustness property."""
    from vnnlib.compat import read_vnnlib_simple
    box, _ = read_vnnlib_simple(vnnlib_path, n_in, n_out)[0]
    box = np.array(box, dtype=np.float64)
    with open(vnnlib_path) as f:
        pairs = re.findall(r"\(>=\s*Y_(\d+)\s+Y_(\d+)\s*\)", f.read())
    labels = {int(c) for _, c in pairs}
    others = {int(j) for j, _ in pairs}
    if len(labels) != 1 or len(others) != n_out - 1:
        raise ValueError("{} is not a robustness property".format(vnnlib_path))
    lower = torch.tensor(box[:, 0], dtype=data_type).to(device)
    upper = torch.tensor(box[:, 1], dtype=data_type).to(device)
    return lower, upper, labels.pop()


def margin(nn, relu_on_output, x, label):
    """max_j o_j - o_c of the network at x by a plain forward pass (float64)."""
    with torch.no_grad():
        out = nn.double()(torch.as_tensor(x, dtype=torch.float64).unsqueeze(0)).reshape(-1)
        nn.to(data_type)
    if relu_on_output:
        out = torch.relu(out)
    others = torch.cat([out[:label], out[label + 1:]])
    return (others.max() - out[label]).item()


# ---------------------------------------------------------------------------------------------------------------
# bounds

def relu_net(nn, relu_on_output):
    """The ONNX network as a plain torch.nn.Sequential for auto_LiRPA (without the identity layer)."""
    linears = [m for m in nn if isinstance(m, torch.nn.Linear)]
    if relu_on_output:
        linears = linears[:-1]
    modules = []
    for k, lin in enumerate(linears):
        layer = torch.nn.Linear(lin.in_features, lin.out_features, dtype=data_type)
        layer.weight.data = lin.weight.data.clone()
        layer.bias.data = lin.bias.data.clone()
        modules.append(layer)
        if k < len(linears) - 1 or relu_on_output:
            modules.append(torch.nn.ReLU())
    return torch.nn.Sequential(*modules)


def initial_bounds(nn, relu_on_output, lower, upper, method):
    """
    (bounds_affine_out, bounds_layer_out) in the layout MultiDHOV expects, one [lb, ub] per SequentialNN layer:
    pre-activation bounds, and after ReLU for every layer but the last. method: "ibp" (the old pipeline's own
    interval propagation), "crown" or "alpha-crown" (auto_LiRPA; the pre-activation bounds are the bounds of the
    inputs of its ReLU nodes).
    """
    if method == "ibp":
        affine, layer_out = nn.calculate_box_bounds([lower, upper])
        return [[lb.detach(), ub.detach()] for lb, ub in affine], [[lb.detach(), ub.detach()] for lb, ub in layer_out]

    from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
    from auto_LiRPA.operators import BoundRelu
    net = relu_net(nn, relu_on_output)
    bounded = BoundedModule(net, torch.zeros((1, lower.numel()), dtype=data_type))
    x = BoundedTensor(((lower + upper) / 2).unsqueeze(0), PerturbationLpNorm(x_L=lower.unsqueeze(0),
                                                                             x_U=upper.unsqueeze(0)))
    with quiet():
        out_lb, out_ub = bounded.compute_bounds(x=(x,), method=method)
    pre = []
    for node in bounded.nodes():
        if isinstance(node, BoundRelu):
            pre.append([node.inputs[0].lower.detach().reshape(-1).to(data_type),
                        node.inputs[0].upper.detach().reshape(-1).to(data_type)])
    if not relu_on_output:
        pre.append([out_lb.detach().reshape(-1).to(data_type), out_ub.detach().reshape(-1).to(data_type)])
    n_layers = len([m for m in nn if isinstance(m, torch.nn.Linear)])
    if relu_on_output:
        # the identity layer's input is the ReLU of the logits
        pre.append([torch.relu(pre[-1][0]), torch.relu(pre[-1][1])])
    if len(pre) != n_layers:
        raise RuntimeError("got bounds of {} layers from auto_LiRPA, expected {}".format(len(pre), n_layers))
    layer_out = [[torch.relu(lb), torch.relu(ub)] for lb, ub in pre[:-1]] + [pre[-1]]
    return pre, layer_out


def check_bounds(nn, lower, upper, bounds_affine_out, samples=200):
    """Raises if a sampled input (and the box corners of the centre) leaves the pre-activation bounds."""
    x = lower + (upper - lower) * torch.rand((samples, lower.numel()), dtype=data_type)
    x = torch.cat([x, ((lower + upper) / 2).unsqueeze(0), lower.unsqueeze(0), upper.unsqueeze(0)])
    linears = [m for m in nn if isinstance(m, torch.nn.Linear)]
    with torch.no_grad():
        for k, lin in enumerate(linears):
            x = lin(x)
            lb, ub = bounds_affine_out[k]
            slack = 1e-4 * (1 + x.abs())
            if ((x < lb - slack) | (x > ub + slack)).any():
                raise RuntimeError("sampled points leave the initial bounds of layer {}".format(k))
            x = torch.relu(x)


def precheck(nn, relu_on_output, lower, upper, label, method, n_out):
    """The bound method's upper bound of max_j o_j - o_c (auto_LiRPA with a spec matrix), or None for ibp."""
    if method == "ibp":
        return None
    from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
    net = relu_net(nn, relu_on_output)
    bounded = BoundedModule(net, torch.zeros((1, lower.numel()), dtype=data_type))
    x = BoundedTensor(((lower + upper) / 2).unsqueeze(0), PerturbationLpNorm(x_L=lower.unsqueeze(0),
                                                                             x_U=upper.unsqueeze(0)))
    c = torch.zeros((1, n_out - 1, n_out), dtype=data_type)
    for row, j in enumerate(j for j in range(n_out) if j != label):
        c[0, row, j] = 1
        c[0, row, label] = -1
    with quiet():
        _, ub = bounded.compute_bounds(x=(x,), C=c, method=method)
    return verbas.widen_up(ub.max().item())


# ---------------------------------------------------------------------------------------------------------------
# property problem

def solve_property(dhov_verifier, nn, relu_on_output, label, n_out, output_layer_index, args, time_limit):
    """
    max_j o_j - o_c over the relaxation (one MILP with a max constraint, as the notebook's add_output_constraints).
    Returns a dict: verdict, bound (dual bound), o_value (bound widened), primal, status, gap, secs, input.
    """
    with grp.Env(params=get_gurobi_params()) as env, dhov_verifier.nn_encoding_model.copy(env) as model:
        model.Params.Threads = args.threads or os.cpu_count()
        model.Params.TimeLimit = max(time_limit, 1.0)
        model.Params.MIPGap = args.property_gap
        if args.stop == "decided":
            # as dhov_rust's [property] stop = "decided": stop at the first proven bound that verifies after widening
            # (the threshold of dhov_rust's Tolerance), or at the first relaxation point with max_j o_j - o_c > 0.
            # One MILP over the maximum, so this also ends "the other classes" at once.
            model.Params.BestBdStop = -solver_tolerance_absolute / (1 - solver_tolerance_relative) - 1e-8
            model.Params.BestObjStop = 1e-9
        input_vars = grp.MVar.fromlist([model.getVarByName("output_layer_[-1]_[{}]".format(k))
                                        for k in range(nn[0].in_features)])
        out = [model.getVarByName("output_layer_[{}]_[{}]".format(output_layer_index, k)) for k in range(n_out)]
        out_lb = dhov_verifier.bounds_layer_out[-1][0].detach().cpu().numpy().astype(np.float64)
        out_ub = dhov_verifier.bounds_layer_out[-1][1].detach().cpu().numpy().astype(np.float64)
        others = [j for j in range(n_out) if j != label]
        diff_lb = [verbas.widen_down(out_lb[j] - out_ub[label]) for j in others]
        diff_ub = [verbas.widen_up(out_ub[j] - out_lb[label]) for j in others]
        diff = [model.addVar(lb=lb, ub=ub, name="diff_var_{}".format(j)) for lb, ub, j in zip(diff_lb, diff_ub, others)]
        for d, j in zip(diff, others):
            model.addConstr(d == out[j] - out[label])
        max_var = model.addVar(lb=min(diff_lb), ub=max(diff_ub), name="max_var")
        model.addConstr(max_var == grp.max_(diff))
        model.setObjective(max_var, grp.GRB.MAXIMIZE)
        start = time.time()
        model.optimize()
        secs = time.time() - start
        if model.Status == grp.GRB.INFEASIBLE:
            raise RuntimeError("property problem infeasible")
        bound = verbas.proven_bound(model)
        primal = model.ObjVal if model.SolCount > 0 else None
        candidate = np.array(input_vars.X) if model.SolCount > 0 else None
        status = {grp.GRB.OPTIMAL: "optimal", grp.GRB.TIME_LIMIT: "time_limit",
                  grp.GRB.USER_OBJ_LIMIT: "stopped"}.get(model.Status, str(model.Status))
        gap = model.MIPGap if model.SolCount > 0 and model.IsMIP else None
    o_value = verbas.widen_up(bound) if bound is not None else None
    verdict = "unknown"
    real_margin = None
    if o_value is not None and o_value < 0:
        verdict = "verified"
    elif candidate is not None:
        real_margin = margin(nn, relu_on_output, candidate, label)
        if real_margin >= 0:
            verdict = "falsified"
    if verdict == "unknown" and status == "time_limit":
        verdict = "timeout"
    if status == "stopped" and verdict != "verified":
        # stopped by the early stop at a point above 0: the bound only says "not verifiable", as dhov_rust, whose
        # o_value is empty when the early stop cancelled classes (the raw dual bound stays in "bound")
        o_value = None
    return {"verdict": verdict, "bound": bound, "o_value": o_value, "primal": primal, "status": status, "gap": gap,
            "secs": secs, "margin_at_incumbent": real_margin}


# ---------------------------------------------------------------------------------------------------------------
# one instance

def run_instance(net_path, spec_path, timeout, args):
    started = time.time()
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    nn, relu_on_output = build_network(net_path)
    n_in = nn[0].in_features
    n_out = [m for m in nn if isinstance(m, torch.nn.Linear)][-1].out_features
    lower, upper, label = read_spec(spec_path, n_in, n_out)
    center = (lower + upper) / 2
    record = {"net": os.path.basename(net_path), "spec": os.path.basename(spec_path), "config": args.config_name,
              "label": label, "seed": seed}

    centre_margin = margin(nn, relu_on_output, center, label)
    if centre_margin >= 0:
        record.update(verdict="falsified", bound=None, o_value=None, precheck=None, layers=[], property=None,
                      falsified_at="centre", times={"total": time.time() - started})
        return record

    t = time.time()
    bounds_affine_out, bounds_layer_out = initial_bounds(nn, relu_on_output, lower, upper, args.bounds)
    check_bounds(nn, lower, upper, bounds_affine_out)
    initial_secs = time.time() - t

    record["precheck"] = None
    if args.precheck:
        t = time.time()
        pre = precheck(nn, relu_on_output, lower, upper, label, args.bounds, n_out)
        record["precheck"] = {"method": args.bounds, "bound": pre, "verified": pre is not None and pre < 0,
                              "secs": time.time() - t}
        if record["precheck"]["verified"]:
            record.update(verdict="verified", bound=pre, o_value=pre, layers=[], property=None,
                          times={"initial_bounds": initial_secs, "relaxation": 0.0, "property": 0.0,
                                 "total": time.time() - started})
            return record

    n_relu_layers = len([m for m in nn if isinstance(m, torch.nn.Linear)]) - 1
    last_relu = n_relu_layers - 1
    if args.variant == "snr":
        layers_as_snr, layers_as_milp = list(range(n_relu_layers)), []
    elif args.variant == "snr_milp":
        layers_as_snr, layers_as_milp = list(range(last_relu)), [last_relu]
    else:
        layers_as_snr, layers_as_milp = [], [last_relu]

    sampling_strategy = PerGroupLineSearchSamplingStrategy(center, [lower, upper], nn, sample_count=args.sample_count,
                                                           num_iterations=args.iterations,
                                                           rand_samples_percent=args.noise_samples,
                                                           rand_sample_alternation_percent=args.noise_coordinates)
    icnn_factory = ICNNFactory("logical", args.icnn_hidden + [1], always_use_logical_layer=False)
    dhov_verifier = multidhov.MultiDHOV()
    t = time.time()
    output = contextlib.nullcontext() if args.verbose else quiet()
    with output:
        finished = dhov_verifier.start_verification(
            nn, center, icnn_factory, args.group_size, [lower, upper], sampling_strategy,
            init_affine_bounds=bounds_affine_out, init_layer_bounds=bounds_layer_out,
            skip_last_layer=relu_on_output, icnn_epochs=args.epochs, icnn_batch_size=10000,
            use_over_approximation=True, tighten_bounds=True, use_fixed_neurons_in_grouping=False,
            layers_as_snr=layers_as_snr, layers_as_milp=layers_as_milp,
            force_inclusion_steps=args.force_inclusion_steps, preemptive_stop=True, grouping_method="consecutive",
            optimizer=args.optimizer, learning_rate=args.learning_rate, init_network=True, adapt_lambda="included",
            encode_icnn_enlargement_as_lp=False, encode_relu_enlargement_as_lp=False,
            time_out=timeout - (time.time() - started), time_per_neuron_refinement=args.bound_time_limit,
            time_per_icnn_refinement=args.enlarge_time_limit,
            allow_heuristic_timeout_estimate=args.heuristic_timeout)
    relaxation_secs = time.time() - t
    record["layers"] = dhov_verifier.layer_reports
    times = {"initial_bounds": initial_secs, "relaxation": relaxation_secs}
    if not finished:
        record.update(verdict="timeout", bound=None, o_value=None, property=None,
                      times=dict(times, property=0.0, total=time.time() - started))
        return record

    output_layer_index = n_relu_layers - 1 if relu_on_output else n_relu_layers
    remaining = timeout - (time.time() - started)
    prop = solve_property(dhov_verifier, nn, relu_on_output, label, n_out, output_layer_index, args,
                          min(remaining, args.property_time_limit))
    record.update(verdict=prop["verdict"], bound=prop["bound"], o_value=prop["o_value"], property=prop,
                  times=dict(times, property=prop["secs"], total=time.time() - started))
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variant", choices=["snr", "snr_milp", "dhov"], required=True)
    parser.add_argument("--bounds", choices=["ibp", "crown", "alpha-crown"], default="alpha-crown")
    parser.add_argument("--instances", required=True, help="VNN-COMP style instances.csv (net,spec,timeout)")
    parser.add_argument("--nets", help="directory of the networks (default: next to instances.csv)")
    parser.add_argument("--out", required=True, help="JSON lines output file (appended to)")
    parser.add_argument("--limit", type=int, help="only the first N instances")
    parser.add_argument("--only", nargs="*", help="only these spec file names")
    parser.add_argument("--config-name", help="value of the config field (default: python_<variant>_<bounds>)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=0, help="Gurobi threads of the property MILP (0: all cores)")
    parser.add_argument("--stop", choices=["decided", "optimal"], default="decided",
                        help="decided (default, as dhov_rust's shipped configs): the property MILP stops once the sign "
                             "is known, so only the sign of the o-value is meaningful; optimal: exact o-values")
    parser.add_argument("--precheck", action=argparse.BooleanOptionalAction, default=True,
                        help="verify by the bound method first, as dhov_rust's [property] precheck (default on; "
                             "--no-precheck to skip it)")
    parser.add_argument("--property-gap", type=float, default=0.0, help="MIPGap of the property MILP")
    parser.add_argument("--property-time-limit", type=float, default=600)
    parser.add_argument("--bound-time-limit", type=float, default=60, help="time limit of each bound LP")
    parser.add_argument("--enlarge-time-limit", type=float, default=50, help="time limit of each enlargement MILP")
    parser.add_argument("--heuristic-timeout", action="store_true",
                        help="the notebook's time-out estimate (dhov_rust has none)")
    # DHOV settings of dhov_rust's configs/dhov/abstract_dhov.toml
    parser.add_argument("--group-size", type=int, default=20)
    parser.add_argument("--sample-count", type=int, default=3000)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--noise-samples", type=float, default=0.2)
    parser.add_argument("--noise-coordinates", type=float, default=0.01)
    parser.add_argument("--icnn-hidden", type=int, nargs="+", default=[5])
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--force-inclusion-steps", type=int, default=3)
    parser.add_argument("--optimizer", choices=["adam", "SdLBFGS"], default="adam")
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--verbose", action="store_true", help="show the pipeline's prints")
    args = parser.parse_args()
    args.config_name = args.config_name or "python_{}_{}".format(args.variant, args.bounds.replace("-", "_"))

    instances_dir = os.path.dirname(os.path.abspath(args.instances))
    nets_dir = args.nets or instances_dir
    with open(args.instances) as f:
        rows = [row for row in csv.reader(f) if row]
    if args.only:
        rows = [row for row in rows if row[1] in args.only]
    if args.limit is not None:
        rows = rows[:args.limit]

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    checked_nets = set()
    for net, spec, timeout in rows:
        net_path = os.path.join(nets_dir, net)
        spec_path = os.path.join(instances_dir, spec)
        if net_path not in checked_nets:
            nn, relu_on_output = build_network(net_path)
            probe = np.random.default_rng(0).random((8, nn[0].in_features))
            check_against_onnxruntime(net_path, nn, relu_on_output, probe)
            checked_nets.add(net_path)
        print("{} {} ...".format(args.config_name, spec), flush=True)
        try:
            record = run_instance(net_path, spec_path, float(timeout), args)
        except Exception as error:  # one failing instance must not stop the suite
            record = {"net": net, "spec": spec, "config": args.config_name, "verdict": "error",
                      "error": "{}: {}".format(type(error).__name__, error), "traceback": traceback.format_exc()}
        print("    {} o_value={} total={:.1f}s".format(record["verdict"], record.get("o_value"),
                                                       record.get("times", {}).get("total", math.nan)), flush=True)
        with open(args.out, "a") as f:
            f.write(json.dumps(record, default=float) + "\n")


if __name__ == "__main__":
    main()
