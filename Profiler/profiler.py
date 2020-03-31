import gc
import torch
from datetime import datetime
import os

OUTPUT_DIR="./memory_csv_data/"

class memory_profiler:
    def __init__(self,model,print_period=1):
        """
        Arguments:
            print_period:
                Integer >=1
        """
        if print_period<1:
            print("ERROR: Please use a print period larger than zero!")
            exit()

        self.model=model
        self.activation_data_pointers=set()
        self.memory_used_by_feature_maps=0
        self.gradient_data_pointers=set()
        self.memory_used_by_gradients=0

        # Print stats every period of iterations
        self.print_period=print_period
        self.iteration=0
        self.epoch=1

        # Gather the named parameters of the model (i.e. layers)
        self.gather_named_parameters()

        # Feature maps:
        for name, layer in self.model._modules.items():
            layer.register_forward_hook(self.forward_hook)
            self.recursive_hooks(layer)

        # Register hooks for gradients
        self.model.register_backward_hook(self.backwards_hook)

        # Initialize the output directory
        # TODO
        try:
            if not os.path.exists(OUTPUT_DIR):
                os.mkdir(OUTPUT_DIR)
        except OSError:
            print(f"Creation of the output directory {OUTPUT_DIR} failed")

        # Initialize the output .csv file
        fname=str(datetime.now().strftime("%Y%m%d%H%M%S")) + ".csv"
        with open(OUTPUT_DIR+fname,"w") as file:
            file.write("ORDER?")
        

    def gather_named_parameters(self):
        """
        Gathers named_parameters from the model.
        """
        self.params={} # data_pointer -> {"tensor","size","name"}
        for name,param in self.model.named_parameters():
            dp=getDataPtr(param)
            self.params[dp]={}
            self.params[dp]["tensor"]=param # The actual tensor
            self.params[dp]["size"]=getTensorSize(param,scale="B")
            self.params[dp]["name"]=name #user-specified name
            self.params[dp]["grad_size"]=0

    def total_layer_mem_MB(self):
        """
        Calculates the total memory usage of all the named parameters in self.params
        """
        r=0
        for dp in self.params:
            r+=self.params[dp]["size"]
        return MB(r)

    def recursive_hooks(self, layer):
        for name, layer in layer._modules.items():
            # We recursively register hooks on all layers
            layer.register_forward_hook(self.forward_hook)
            self.recursive_hooks(layer)

    def forward_hook(self,m, i, o):
        '''
        The hook function to be registered on each module

        Parameters:
            m: module (i.e. layer)
            i: tuple of input activation tensors
            o: output activation tensor
        '''
        if getDataPtr(o) not in self.activation_data_pointers:
            self.activation_data_pointers.add(getDataPtr(o))
            self.memory_used_by_feature_maps+=getTensorSize(o,scale="B")

    def backwards_hook(self, m, in_grads, out_grads):
        """
        Registers hooks for the backward pass. By calling this as:
            model.register_backward_hook(self.backwards_hook)
        all intermediate gradients are registered as well, and not just
        the gradients of leaf nodes in the computational graph.
        https://discuss.pytorch.org/t/using-hook-function-to-save-gradients/4334

        Intermediate gradient tensors are accumulated in C++ buffers, and are only exposed to Python by using hooks.
        https://discuss.pytorch.org/t/how-the-hook-works/2222

        Parameters:
            m: module
            in_grads: input gradients
            out_grads: output gradients

        """
        # First, inspect the .grad of each module parameter
        for dp in self.params:
            t=self.params[dp]['tensor']
            if t.grad is not None and getDataPtr(t.grad) not in self.gradient_data_pointers:
                self.gradient_data_pointers.add(getDataPtr(t.grad))
                self.memory_used_by_gradients+=getTensorSize(t.grad,scale="B")
                self.params[dp]["grad_size"]+=getTensorSize(t.grad,scale="B")
                #print(f"Caugt a .grad! {self.params[dp]['name']}.grad is size {self.params[dp]['grad_size']}")

        for t in in_grads:
            if getDataPtr(t) not in self.gradient_data_pointers:
                self.gradient_data_pointers.add(getDataPtr(t))
                self.memory_used_by_gradients+=getTensorSize(t,scale="B")

        for t in out_grads:
            if getDataPtr(t) not in self.gradient_data_pointers:
                self.gradient_data_pointers.add(getDataPtr(t))
                self.memory_used_by_gradients+=getTensorSize(t,scale="B")
        


    def record_stats(self):
        """
        Record statistics about memory usage for this iteration.

        If print_period iterations have passed, also print 
        diagnostics about memory usage.

        Note: torch.cuda.max_memory_cached() and torch.cuda.memory_cached() are deprecated in later versions, replaced by torch.cuda.max_memory_reserved() and torch.cuda.memory_reserved().
        """
        self.iteration+=1
        
        if self.iteration % self.print_period == 0:
            self.print_info_table()
        
        # # Reset computed memories after each iteration
        # self.activation_data_pointers=set()
        # self.memory_used_by_feature_maps=0
        # self.gradient_data_pointers=set()
        # self.memory_used_by_gradients=0


    def print_info_single_line(self):
        s=""
        for dp in self.params:
            s+=self.params[dp]["name"]+"="+str(MB(self.params[dp]["size"]))+", "

        s+=f"Peak allocated={MB(torch.cuda.max_memory_allocated())}, "
        #s+=f"Current allocated={MB(torch.cuda.memory_allocated())}, "
        s+=f"Peak cached={MB(torch.cuda.max_memory_cached())}, "
        s+=f"Current cached={MB(torch.cuda.memory_cached())}, "
        s+=f"Layer usage={self.total_layer_mem_MB()}, "
        s+=f"Activation usage={MB(self.memory_used_by_feature_maps)}, "
        s+=f"Gradient usage={MB(self.memory_used_by_gradients)}"
        print(s)
    
    def print_info_table(self):
        """
        Print memory diagnostics about this iteration in a user-friendly table
        """
        
        dash = '-' * 44
        print("\n"+dash)
        print("Memory Usage for Iteration",self.iteration, "of epoch",self.epoch)
        print(dash)

        print('{:.<35s}{:.>5d} MB'.format("Peak allocated", MB(torch.cuda.max_memory_allocated())))
        #print('{:.<35s}{:.>5d} MB'.format("Current allocated", MB(torch.cuda.memory_allocated())))
        print('{:.<35s}{:.>5d} MB'.format("Peak cached", MB(torch.cuda.max_memory_cached())))
        print('{:.<35s}{:.>5d} MB'.format("Current cached", MB(torch.cuda.memory_cached())))
        print('{:.<35s}{:.>5d} MB'.format("Activation usage", MB(self.memory_used_by_feature_maps)))
        
        # Layer-by-layer weight breakdown
        print('\n{:.<35s}{:.>5d} MB'.format("Total weight usage", self.total_layer_mem_MB()))
        for dp in self.params:
            print('{:<2s}{:.<33s}{:.>5d} MB'.format('',self.params[dp]["name"], MB(self.params[dp]["size"])))

        # Layer-by-layer gradient breakdown
        unnamed_gradient_mem=self.memory_used_by_gradients
        print('\n{:.<35s}{:.>5d} MB'.format("Total gradient usage", MB(self.memory_used_by_gradients)))
        for dp in self.params:
            print('{:<2s}{:.<33s}{:.>5d} MB'.format('',self.params[dp]["name"] + " grad", MB(self.params[dp]["grad_size"])))
            unnamed_gradient_mem-=self.params[dp]["grad_size"]
        print('{:<2s}{:.<33s}{:.>5d} MB'.format('',"Intermediate grads", MB(unnamed_gradient_mem)))
            
    
    def epoch_end(self):
        print(f"Epoch {self.epoch} finished")
        self.activation_data_pointers=set()
        self.memory_used_by_feature_maps=0
        self.gradient_data_pointers=set()
        self.memory_used_by_gradients=0
        self.iteration=0
        self.epoch+=1
        for dp in self.params:
            self.params[dp]["grad_size"]=0


def getDataPtr(tensor):
    """
    Get the data pointer of a tensor. The data pointer is used
    to uniquely identify a tensor in memory.
    """
    return tensor.storage().data_ptr()

def getTensorSize(tensor, scale="MB"):
    """
    Get the size of a tensor
    """
    element_size = tensor.element_size()
    numel = tensor.storage().size()
    memory_size = numel * element_size
    if scale=="MB":
        return MB(memory_size)
    if scale=="B":
        return memory_size

def MB(B):
    """
    Convert Bytes to MB
    """
    return B//1000000 