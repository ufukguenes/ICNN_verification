
import os
import sys
import contextlib
import argparse

# set to ICNN_verification root
sys.path.append('../../..')

import torch
import time
from script.NeuralNets.Networks import SequentialNN
from script.settings import device, data_type
import script.DHOV.MultiDHOV as multidhov
import gurobipy as grp
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize
from script.NeuralNets.ICNNFactory import ICNNFactory
import matplotlib.pyplot as plt
from script.Profiling import Timings
import numpy as np
import csv
from script.DHOV.Sampling.PerGroupLineSearchSampling import PerGroupLineSearchSamplingStrategy
from script.DHOV.Sampling.PerGroupSamplingStrategy import PerGroupSamplingStrategy
from script.DHOV.Sampling.PerGroupFeasibleSamplingStrategy import PerGroupFeasibleSamplingStrategy
from script.DHOV.Sampling.ZonotopeSamplingStrategy import ZonotopeSamplingStrategy

import onnx
from onnx2pytorch import ConvertModel
from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm

from vnnlib.compat import read_vnnlib_simple
from collections import OrderedDict




def imshow(img):
    img = img / 2 + .05  # revert normalization for viewing
    npimg = img.to("cpu").numpy()
    plt.imshow(npimg, cmap="gray")
    plt.show()


def accuracy_test(model, converted_model, data):
    total_correct = 0
    total_wrong = 0
    for image, label in data:
        test_image = torch.unsqueeze(image, 0).to(dtype=data_type).to(device)   
        model_test = model(test_image)
        converted_test = converted_model(test_image)
        if not torch.isclose(model_test, converted_test).all():
            print("is not close")
            break
        if torch.argmax(model_test).item() == label:
            total_correct += 1
        else:
            total_wrong += 1
    
    print("is close")    
    print(f"accuracy {total_correct/ len(data)}")


def add_output_constraints(model, nn_layer_out_bounds, label, output_vars, sovler_bound=1e-3):
    """
    
    :param model: the optimization problem in gurobi encoding the NN
    :param nn_layer_out_bounds: torch.Tensor, approximating the upper and lower bounding the output layer of the NN
    :param label: index of the label or target neuron which is compared against
    :param output_vars: the gurobi variables from the model of the NN describing the output neurons of the NN
    :param sovler_bound: provides a bound for the gurobi solver. If this bound is achieved, the optimizer stops
    """
    
    out_lb = nn_layer_out_bounds[-1][0].detach().cpu().numpy()
    out_ub = nn_layer_out_bounds[-1][1].detach().cpu().numpy()
    
    difference_lb = out_lb - out_ub[label]
    difference_ub = out_ub - out_lb[label]
    difference_lb = difference_lb.tolist()
    difference_ub = difference_ub.tolist()
    
    difference_lb.pop(label)
    difference_ub.pop(label)
    
    min_diff = min(difference_lb)
    max_diff = max(difference_ub)
    
    difference = model.addVars(9, lb=difference_lb, ub=difference_ub, name="diff_var")
    model.addConstrs((difference[i] == output_vars.tolist()[i] - output_vars.tolist()[label] for i in range(0, label)), name="diff_const0")
    model.addConstrs((difference[i - 1] == output_vars.tolist()[i] - output_vars.tolist()[label] for i in range(label + 1, 10)), name="diff_const1")

    max_var = model.addVar(lb=min_diff, ub=max_diff, name="max_var")
    model.addConstr(max_var == grp.max_(difference))

    if sovler_bound != None:
        model.setParam("BestObjStop", sovler_bound)

    model.update()
    model.setObjective(max_var, grp.GRB.MAXIMIZE)
    

def get_output_vars_dhov(model, output_size, output_layer_index):
    output_vars = []
    for i in range(output_size):
        output_vars.append(model.getVarByName("output_layer_[{}]_[{}]".format(output_layer_index, i)))
    output_vars = grp.MVar.fromlist(output_vars)
    return output_vars


def optimize_model(model, output_vars, start_overall_time, dhov_timings, time_limit=60*60, verbose=True, csv_to_write_to=None, csv_row_name="No name"):
    """
    
    :param model: the optimization problem in gurobi encoding the NN and the objective 
    :param output_vars: the gurobi variables from the model of the NN describing the output neurons of the NN
    :return True if verification was successful, else false 
    """
    
    model.setParam("TimeLimit", time_limit)
    
    start_solving_time = time.time()
    model.update()
    model.optimize()
    
    end_time = time.time()
    time_just_solving = end_time - start_solving_time
    time_overall = end_time - start_overall_time
    
    
    if verbose:
        print("time for verification {}".format(time_just_solving))
        print("overall time {}".format(time_overall))
    
    if model.Status == grp.GRB.OPTIMAL or model.Status == grp.GRB.USER_OBJ_LIMIT:
                    
        max_var = model.getVarByName("max_var").getAttr("x")
        verification_successful = max_var < 0
        
        if csv_to_write_to is not None:
            new_row = [csv_row_name, "finished", time_just_solving, time_overall, verification_successful, max_var] +  dhov_timings.get_all_results(do_round=True)
            
            with open(csv_to_write_to, 'a', newline='') as file_obj:
                writer_object = csv.writer(file_obj)
             
                writer_object.writerow(new_row)
                file_obj.close()
        
        if verbose:
            for i, var in enumerate(output_vars.tolist()):
                print("var {}: {}".format(i, var.getAttr("x")))
                
            if verification_successful:
                print("property verified with max difference {}".format(max_var))
                return True
            else:
                 print("property NOT verified with max difference {}".format(max_var))
                
                
        
        
        return verification_successful
    
    elif model.Status == grp.GRB.TIME_LIMIT:        
        
        max_var_upper_bound = model.getAttr("ObjBound")
        
        verification_failed_with_upper_bound = max_var_upper_bound > 0
        
        if verbose:
            if verification_failed_with_upper_bound:
                print("property NOT verified with upper bound for max difference {}".format(max_var_upper_bound))
            else:
                print("time out and upper bound could not disprove the setting")
            
        if csv_to_write_to is not None:
            new_row = [csv_row_name, "time_out", time_just_solving, time_overall, verification_failed_with_upper_bound, max_var_upper_bound] + dhov_timings.get_all_results()
            with open(csv_to_write_to, 'a', newline='') as file_obj:
                writer_object = csv.writer(file_obj)
             
                writer_object.writerow(new_row)
                file_obj.close()
                
                
        return verification_failed_with_upper_bound

    elif model.Status == grp.GRB.INFEASIBLE:
        
        if csv_to_write_to is not None:
            new_row = [csv_row_name, "infeasible", time_just_solving, time_overall, False,]
            with open(csv_to_write_to, 'a', newline='') as file_obj:
                writer_object = csv.writer(file_obj)
             
                writer_object.writerow(new_row)
                file_obj.close()
        
        if verbose:
            print("model infeasible")
    
            model.computeIIS()
            print("constraint")
            all_constr = model.getConstrs()
    
            for const in all_constr:
                if const.IISConstr:
                    print("{}".format(const))
    
            print("lower bound")
            all_var = model.getVars()
            for var in all_var:
                if var.IISLB:
                    print("{}, lb: {}, ub: {}".format(var, var.getAttr("lb"), var.getAttr("ub")))
    
            print("upper bound")
            all_var = model.getVars()
            for var in all_var:
                if var.IISUB:
                    print("{}, lb: {}, ub: {}".format(var, var.getAttr("lb"), var.getAttr("ub")))
                
    return False



def load_vnnlib_bounds(vnnlib_path, input_shape, n_out):
    n_in = np.prod(input_shape)
    res = read_vnnlib_simple(vnnlib_path, n_in, n_out)
    bnds, spec = res[0]
    
    bnds = np.array(bnds)
    lbs = bnds[:,0]
    ubs = bnds[:,1]
    
    data_min = torch.tensor(lbs, dtype=data_type).reshape(input_shape).to(device)
    data_max = torch.tensor(ubs, dtype=data_type).reshape(input_shape).to(device)

    return [data_min, data_max]


def onnx_to_bounded_model(onnx_path, input_shape):
    onnx_model = onnx.load(onnx_path)
    torch_model = ConvertModel(onnx_model)
    
    x_concrete = torch.zeros(input_shape)
    model = BoundedModule(torch_model, x_concrete)
    return model



def load_vnnlib_spec_for_auto_lirpa(vnnlib_path, input_shape, n_out):
    n_in = np.prod(input_shape)
    res = read_vnnlib_simple(vnnlib_path, n_in, n_out)
    bnds, spec = res[0]
    
    bnds = np.array(bnds)
    lbs = bnds[:,0]
    ubs = bnds[:,1]
    
    data_min = torch.tensor(lbs, dtype=data_type).reshape(input_shape)
    data_max = torch.tensor(ubs, dtype=data_type).reshape(input_shape)
    center = 0.5*(data_min + data_max)

    ptb = PerturbationLpNorm(x_L=data_min, x_U=data_max)
    x = BoundedTensor(center, ptb)
    
    return x, center



def get_layers(model):
    return [l for l in model.nodes() if l.perturbed]


def get_intermediate_bounds(model):
    """
    Returns a dictionary containing the concrete lower and upper bounds of each layer.
    
    Implemented own method to filter out bounds for weight matrices.
    
    Only call this method after compute_bounds()!
    """
    od = OrderedDict()
    for l in get_layers(model):
        if hasattr(l, 'lower'):
            od[l.name] = (l.lower, l.upper)
            
    return od


def get_bounds_auto_lirpa(x: BoundedTensor, model: BoundedModule, method="crown"):
    model.compute_bounds(x=(x,), method=method, bound_lower=True, bound_upper=True)
    bounds_dict_crown = get_intermediate_bounds(model)
    crown_bounds_affine_out = []
    prev_key = None
    for i, key in enumerate(bounds_dict_crown.keys()):
        if i == 0: # use this if ibp is used (or i % 2 == 1:)
            continue
        elif method == "alpha-crown" and "59" in key:
            # todo WTF, why do i need to do this?
            lb, ub = bounds_dict_crown[key][0], bounds_dict_crown[prev_key][1]
            crown_bounds_affine_out.append([lb.type(data_type).view(-1).to(device), ub.type(data_type).view(-1).to(device)])
        else: 
            lb, ub = bounds_dict_crown[key]
            crown_bounds_affine_out.append([lb.type(data_type).view(-1).to(device), ub.type(data_type).view(-1).to(device)])
        prev_key = key
        
    crown_bounds_layer_out = []
    relu = torch.nn.ReLU()
    for i, (lb, ub) in enumerate(crown_bounds_affine_out):
        if i == len(crown_bounds_affine_out) - 1:
            crown_bounds_layer_out.append([lb, ub])
        else:
            lb_layer = relu(lb)
            ub_layer = relu(ub)
            crown_bounds_layer_out.append([lb_layer, ub_layer])
            
    return crown_bounds_affine_out, crown_bounds_layer_out


def create_csv_file(csv_file_path, settings):
    data = [ ['Input name', 'State of optimization', 'time for just solving', "time overall", "was successful?", "max distance"] +  Timings().get_ordering_as_list_of_strings() + ["settings:", settings] ]
    # File path for the CSV file
    if os.path.isfile(csv_file_path):
        raise RuntimeError("File name {} already exists".format(csv_file_path))
    
    # Open the file in write mode
    with open(csv_file_path, mode='a+', newline='') as file_obj:
        # Create a csv.writer object
        writer_object = csv.writer(file_obj)
        # Write data to the CSV file
        writer_object.writerows(data)
     
    # Print a confirmation message
    print(f"CSV file '{csv_file_path}' created successfully.")


def check_if_image_is_labeled_correctly(nn_model, image, label):
    test_image = torch.unsqueeze(image, 0).to(dtype=data_type).to(device)
    output = nn_model(test_image)
    return torch.argmax(output).item() == label


def check_if_auto_lirpa_already_verifies(model, x, output_size, label, method="crown"):
    c_matrix = torch.zeros((1, output_size - 1, output_size))
    for i in range(output_size - 1):
        if i < label:
            index = i
        elif i >= label:
            index = i + 1
        
        c_matrix[0][i][index] = 1
        c_matrix[0][i][label] = -1
    
    lb, ub = model.compute_bounds(x=(x,), C=c_matrix, method=method)
    
    return ub.max() < 0


def verify_snr_milp_last_layer(vnnlib_dir_path, onnx_path, training_data, nn, cpu_core_count, output_size, number_layer, time_out=60*60, time_per_neuron_refinement=10, verbose=False):
    csv_file_path = "03_snv+milp_mnist_9x200_eps_015.csv"

    crown_method = "alpha-crown"
    tighten_bounds = True
    time_out = 60*60
    time_per_neuron_refinement = 10
    allow_heuristic_timeout_estimate = True
    settings = "crown_method: {}, tighten_bounds: {}, time_out: {}, time_per_neuron_refinement: {}, allow_heuristic_timeout_estimate: {}".format(crown_method, tighten_bounds, time_out, time_per_neuron_refinement, allow_heuristic_timeout_estimate)
    create_csv_file(csv_file_path, settings)


    for num, vnnlib_path in enumerate(sorted(os.listdir(vnnlib_dir_path))): 
        
        if num == 50:
            break
        
        print("{} ================================================".format(num))
        
        full_path = vnnlib_dir_path + "/" + vnnlib_path
        input_bounds = load_vnnlib_bounds(full_path, [784,], 10)
        model = onnx_to_bounded_model(onnx_path, [1,1,1,784])
        image, label = training_data[num]
        x, center = load_vnnlib_spec_for_auto_lirpa(full_path, [1,1,1,784], 10)
        
        if not check_if_image_is_labeled_correctly(nn, image, label):
            print("skipped because wrong classification from the network")
            
            new_row = [vnnlib_path, "wrong classification"]
            with open(csv_file_path, 'a', newline='') as file_obj:
                writer_object = csv.writer(file_obj)
            
                writer_object.writerow(new_row)
                file_obj.close()
                
                
            continue
        
        if check_if_auto_lirpa_already_verifies(model, x, 10, label, method=crown_method):
            print("skipped because auto lirpa already verified")
            
            new_row = [vnnlib_path, "lirpa classified"]
            with open(csv_file_path, 'a', newline='') as file_obj:
                writer_object = csv.writer(file_obj)
            
                writer_object.writerow(new_row)
                file_obj.close()
                
            continue
        
        overall_time = time.time()
        # weirdly the box bounds are faster for snr+milp
        bounds_affine_out, bounds_layer_out = get_bounds_auto_lirpa(x, model, method=crown_method)
        
        # we need to pick these parameters, but for the SNR+MILP case these don't matter
        sampling_strategy = PerGroupLineSearchSamplingStrategy(center, input_bounds, nn, sample_count=100)
        group_size = 20
        net_size = [5, 1]
        icnn_factory = ICNNFactory("logical", net_size, always_use_logical_layer=False)
        
        
        
        dhov_verifier = multidhov.MultiDHOV()
        
        # block prints
        with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
            dhov_verifier.start_verification(nn, center, icnn_factory, group_size, input_bounds, sampling_strategy, 
                                            init_affine_bounds=bounds_affine_out, init_layer_bounds=bounds_layer_out,
                                            skip_last_layer=True,
                                            tighten_bounds=tighten_bounds, 
                                            layers_as_snr=[0, 1, 2, 3, 4, 5, 6, 7], 
                                            layers_as_milp=[8],
                                            time_out=time_out,
                                            time_per_neuron_refinement=time_per_neuron_refinement,
                                            allow_heuristic_timeout_estimate=allow_heuristic_timeout_estimate,)
        

        
        dhov_model = dhov_verifier.nn_encoding_model.copy()
        dhov_model.setParam(grp.GRB.Param.Threads, cpu_core_count)
        dhov_model.update()
        dhov_out_vars = get_output_vars_dhov(dhov_model, output_size, number_layer)
        
        add_output_constraints(dhov_model, dhov_verifier.bounds_layer_out, label, dhov_out_vars)
        
        optimize_model(dhov_model, dhov_out_vars, overall_time, dhov_verifier.timings, verbose=verbose, csv_to_write_to=csv_file_path, csv_row_name=vnnlib_path)


def verify_dhov(vnnlib_dir_path, onnx_path, training_data, nn, cpu_core_count, output_size, number_layer, time_out=60*60, time_per_neuron_refinement=10, verbose=False):
    csv_file_path = "03_dhov_mnist_9x200_eps_015.csv"


    group_size = 20
    sample_count = 3000
    crown_method = "alpha-crown"
    tighten_bounds = True
    time_per_icnn_refinement = 50
    allow_heuristic_timeout_estimate = True
    epochs = 200

    settings = "group_size: {}, sample_count: {}, crown_method: {}, tighten_bounds: {}, time_out: {}, time_per_neuron_refinement: {}, time_per_icnn_refinement: {}, allow_heuristic_timeout_estimate: {}, epochs: {}".format(
        group_size, sample_count, crown_method, tighten_bounds, time_out, time_per_neuron_refinement, time_per_icnn_refinement, allow_heuristic_timeout_estimate, epochs)

    create_csv_file(csv_file_path, settings)
    for num, vnnlib_path in enumerate(sorted(os.listdir(vnnlib_dir_path))):
        
        if num == 50:
            break
            
        print("{} ================================================".format(num))
        
        full_path = vnnlib_dir_path + "/" + vnnlib_path
        input_bounds = load_vnnlib_bounds(full_path, [784,], 10)
        model = onnx_to_bounded_model(onnx_path, [1,1,1,784])
        image, label = training_data[num]
        x, center = load_vnnlib_spec_for_auto_lirpa(full_path, [1,1,1,784], 10)
        
        if not check_if_image_is_labeled_correctly(nn, image, label):
            print("skipped because wrong classification from the network")
            
            new_row = [vnnlib_path, "wrong classification"]
            with open(csv_file_path, 'a', newline='') as file_obj:
                writer_object = csv.writer(file_obj)
            
                writer_object.writerow(new_row)
                file_obj.close()
                
                
            continue
        
        if check_if_auto_lirpa_already_verifies(model, x, 10, label, method=crown_method):
            print("skipped because auto lirpa already verified")
            
            new_row = [vnnlib_path, "lirpa classified"]
            with open(csv_file_path, 'a', newline='') as file_obj:
                writer_object = csv.writer(file_obj)
            
                writer_object.writerow(new_row)
                file_obj.close()
                
            continue
        
        overall_time = time.time()
        
        bounds_affine_out, bounds_layer_out = get_bounds_auto_lirpa(x, model, method=crown_method)
        sampling_strategy = PerGroupLineSearchSamplingStrategy(center, input_bounds, nn, sample_count=sample_count)
        net_size = [5, 1]
        #icnn_factory = ICNNFactory("approx_max", net_size, maximum_function="SMU", function_parameter=0.3)
        icnn_factory = ICNNFactory("logical", net_size, always_use_logical_layer=False)
        #icnn_factory = ICNNFactory("standard", net_size, adapt_layer_for_init=True)
        
        
        dhov_verifier = multidhov.MultiDHOV()
        with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
                return_without_time_out = dhov_verifier.start_verification(nn, center, icnn_factory, group_size, input_bounds, sampling_strategy, 
                                                                        init_affine_bounds=bounds_affine_out, 
                                                                        init_layer_bounds=bounds_layer_out, 
                                                                        skip_last_layer=True,
                                                                        icnn_epochs=epochs, 
                                                                        icnn_batch_size=10000, 
                                                                        use_over_approximation=True, 
                                                                        tighten_bounds=tighten_bounds, 
                                                                        use_fixed_neurons_in_grouping=False, 
                                                                        layers_as_snr=[], 
                                                                        layers_as_milp=[8], 
                                                                        force_inclusion_steps=3, 
                                                                        preemptive_stop=True, 
                                                                        grouping_method="consecutive", 
                                                                        group_num_multiplier=None,
                                                                        optimizer="SdLBFGS", 
                                                                        init_network=True, 
                                                                        adapt_lambda="included",
                                                                        encode_icnn_enlargement_as_lp=False, 
                                                                        encode_relu_enlargement_as_lp=False,
                                                                        time_out=time_out,
                                                                        time_per_neuron_refinement=time_per_neuron_refinement,
                                                                        time_per_icnn_refinement=time_per_icnn_refinement,
                                                                        allow_heuristic_timeout_estimate=allow_heuristic_timeout_estimate,
                                                                        break_after=None, print_training_loss=False, print_new_bounds=False, store_samples=False, print_optimization_steps=False, print_last_loss=False)
        
        if not return_without_time_out:
            new_row = [vnnlib_path, "time_out of DHOV", "-", time.time() - overall_time, False, "-"] +  dhov_verifier.timings.get_all_results(do_round=True)
            with open(csv_file_path, 'a', newline='') as file_obj:
                writer_object = csv.writer(file_obj)
            
                writer_object.writerow(new_row)
                file_obj.close()
            continue
            
        dhov_model = dhov_verifier.nn_encoding_model.copy()
        dhov_model.setParam(grp.GRB.Param.Threads, cpu_core_count)
        dhov_model.update()
        dhov_out_vars = get_output_vars_dhov(dhov_model, output_size, number_layer)
        
        add_output_constraints(dhov_model, dhov_verifier.bounds_layer_out, label, dhov_out_vars)
        
        optimize_model(dhov_model, dhov_out_vars, overall_time, dhov_verifier.timings, verbose=False, csv_to_write_to=csv_file_path, csv_row_name=vnnlib_path)



def main():
    parser = argparse.ArgumentParser(description='Run experiment for DHOV vs LP verification.')
    parser.add_argument('-to', '--timeout', help='timeout per property in seconds', default=60*60)
    parser.add_argument('-tnr', '--time_neuron_refinement', help='timeout per neuron bounds refinement in seconds', default=10)
    parser.add_argument('-odhov', '--only_dhov', help='only run DHOV, no LP verification', action='store_true')
    parser.add_argument('-olp', '--only_lp', help='only run LP verification, no DHOV', action='store_true')
    parser.add_argument('-v', '--verbose', help='enable verbose output', action='store_true')
    args = parser.parse_args()

    if args.only_lp and args.only_dhov:
        raise ValueError("only_dhov and only_lp cannot both be set!")
    
    timeout = float(args.timeout)
    timeout_neuron_refinement = float(args.time_neuron_refinement)
    only_dhov = args.only_dhov
    only_lp = args.only_lp
    verbose = args.verbose

    onnx_name = "mnist_relu_9_200.onnx"
    vnnlib_name = "1000_mnist_eps_015"

    onnx_path = 'nets/' + onnx_name
    vnnlib_dir_path = "specs/" + vnnlib_name
    cpu_core_count = os.cpu_count()

    onnx_model = onnx.load(onnx_path)
    pytorch_model = ConvertModel(onnx_model)
    nn = SequentialNN([28 * 28 * 1, 200, 200, 200, 200, 200, 200, 200, 200, 10, 10])


    parameter_list_onnx = list(pytorch_model.parameters())
    parameter_list_sequential = list(nn.parameters())
    for i in range(0, len(parameter_list_onnx), 2):
        parameter_list_sequential[i].data = parameter_list_onnx[i].data
        parameter_list_sequential[i+1].data = parameter_list_onnx[i+1].data
    parameter_list_sequential[-2].data = torch.eye(10, dtype=data_type)
    parameter_list_sequential[-1].data = torch.zeros(10, dtype=data_type)

    parameter_list = list(pytorch_model.parameters()) # don't use nn here as we have added an extra layer
    output_size = 10
    number_layer = (len(parameter_list) - 2) // 2


    transform = Compose([ToTensor()])
    training_data = MNIST(root="../../../mnist", train=True, download=True, transform=transform)

    do_test = False
    if do_test:
        accuracy_test(pytorch_model, nn, training_data)


    if not only_dhov:
        verify_snr_milp_last_layer(vnnlib_dir_path, onnx_path, training_data, nn, cpu_core_count, output_size, number_layer, 
                                   time_out=timeout, time_per_neuron_refinement=timeout_neuron_refinement, verbose=verbose)

    if not only_lp:
        verify_dhov(vnnlib_dir_path, onnx_path, training_data, nn, cpu_core_count, output_size, number_layer, 
                    time_out=timeout, time_per_neuron_refinement=timeout_neuron_refinement, verbose=verbose)



if __name__ == '__main__':
    main()


