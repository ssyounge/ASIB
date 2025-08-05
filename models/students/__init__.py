# models/students/__init__.py

# Import all student models to register them
from . import resnet50_student
from . import resnet101_student  
from . import resnet152_student
from . import shufflenet_v2_student
from . import mobilenet_v2_student
from . import efficientnet_b0_student

__all__ = [
    "resnet50_student",
    "resnet101_student", 
    "resnet152_student",
    "shufflenet_v2_student",
    "mobilenet_v2_student",
    "efficientnet_b0_student",
]
