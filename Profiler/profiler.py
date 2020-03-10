import gc
import torch

class memory_profiler:
    def __init__(self,model):
        self.model=model

        # Register hooks for gradients
        self.model.register_backward_hook(self.backwards_hook)

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

def getDataPtr(tensor):
    """
    Get the data pointer of a tensor. The data pointer is used
    to uniquely identify a tensor in memory.
    """
    return tensor.storage().data_ptr()

def getTensorSize(tensor):
    """
    Get the size of a tensor in MB
    """
    element_size = tensor.element_size()
    numel = tensor.storage().size()
    memory_size = numel * element_size
    return MB(memory_size)

def MB(B):
    """
    Convert Bytes to MB
    """
    return B//1000000 