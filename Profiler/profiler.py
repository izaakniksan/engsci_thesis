import gc
import torch

class memory_profiler:
    def __init__(self,model):
        self.model=model
        self.activation_data_pointers={}

        # Gather the named parameters of the model (i.e. layers)
        self.gather_named_parameters()

        # Feature maps:
        self.memory_used_by_feature_maps=0
        for name, layer in self.model._modules.items():
            layer.register_forward_hook(self.forward_hook)
            self.recursive_hooks(layer)

        # Register hooks for gradients
        self.model.register_backward_hook(self.backwards_hook)

    def gather_named_parameters(self):
        """
        Gathers named_parameters from the model.
        """
        self.params={} # data_pointer -> {"tensor","size","name"}
        for name,param in self.model.named_parameters():
            dp=getDataPtr(param)
            self.params[dp]={}
            self.params[dp]["tensor"]=param # The actual tensor
            self.params[dp]["size"]=getTensorSize(param)
            self.params[dp]["name"]=name #user-specified name


    def total_layer_mem_MB(self):
        """
        Calculates the total memory usage of all the named parameters in self.params
        """
        r=0
        for dp in self.params:
            r+=self.params[dp]["size"]
        return r

    def recursive_hooks(self, layer):
        for name, layer in layer._modules.items():
            # We recursively register hooks on all layers
            layer.register_forward_hook(self.forward_hook)
            self.recursive_hooks(layer)

    def forward_hook(self,m, i, o):
        if getDataPtr(o) not in self.activation_data_pointers:
            self.activation_data_pointers[getDataPtr(o)]=""
            self.memory_used_by_feature_maps+=getTensorSize(o,scale="B")

        #print(f"forward_hook called with o={o}, with size={getTensorSizeMB(o)}")

    def backwards_hook(self, module, in_grads, out_grads):
        """
        Registers hooks for the backward pass. By calling this as:
            model.register_backward_hook(self.backwards_hook)
        all intermediate gradients are registered as well, and not just
        the gradients of leaf nodes in the computational graph.
        https://discuss.pytorch.org/t/using-hook-function-to-save-gradients/4334
        """
        inputs=[] #(data_ptr, mem_size)
        outputs=[] #(data_ptr, mem_size)
        for t in in_grads:
            in_grad_mem=getTensorSize(t)
            in_grad_data_ptr=getDataPtr(t)
            inputs.append((in_grad_data_ptr, in_grad_mem))

        for t in out_grads:
            out_grad_mem=getTensorSize(t)
            out_grad_data_ptr=getDataPtr(t)
            outputs.append((out_grad_data_ptr, out_grad_mem))

        print(f"backwards_hook called inputs={inputs}, outputs={outputs}")

    def print_stats(self):
        """
        Print statistics about memory usage.

        Note: torch.cuda.max_memory_cached() and torch.cuda.memory_cached() are deprecated in later versions, replaced by torch.cuda.max_memory_reserved() and torch.cuda.memory_reserved().
        """
        self.gather_named_parameters() # Re-gather the layers

        print(f"Peak allocated={MB(torch.cuda.max_memory_allocated())}, "
            f"Current allocated={MB(torch.cuda.memory_allocated())}, "
            f"Peak cached={MB(torch.cuda.max_memory_cached())}, "
            f"Current cached={MB(torch.cuda.memory_cached())}, "
            f"Layer usage={self.total_layer_mem_MB()},"
            f"Activation usage={MB(self.memory_used_by_feature_maps)}"
            )
        self.activation_data_pointers={}
        self.memory_used_by_feature_maps=0


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