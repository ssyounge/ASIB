import pytest
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class TestActualDatasetProblem:
    """Test to identify the actual dataset problem"""
    
    def test_what_dataset_is_actually_loaded(self):
        """Test what type of dataset is actually loaded and why it fails"""
        
        # Let's create a simple test to see what's happening
        def simulate_main_py_dataset_check(dataset):
            """Simulate the exact logic from main.py"""
            print(f"Dataset type: {type(dataset)}")
            print(f"Dataset attributes: {dir(dataset)}")
            
            if isinstance(dataset, torch.utils.data.ConcatDataset):
                print("It's a ConcatDataset, returning 100")
                return 100  # CIFAR-100 fixed
            else:
                print("It's not a ConcatDataset, checking for classes/num_classes")
                n_classes = getattr(dataset, "classes", None)
                print(f"classes attribute: {n_classes}")
                if n_classes is None:
                    n_classes = getattr(dataset, "num_classes", None)
                    print(f"num_classes attribute: {n_classes}")
                if n_classes is None:
                    print("ERROR: Neither classes nor num_classes found!")
                    raise AttributeError("Dataset must expose `classes` or `num_classes`")
                result = len(n_classes) if not isinstance(n_classes, int) else n_classes
                print(f"Result: {result}")
                return result
        
        # Test with a simple dataset that has the attributes
        class GoodDataset(Dataset):
            def __init__(self):
                self.classes = list(range(100))
                self.num_classes = 100
                self.data = torch.randn(100, 3, 32, 32)
                self.labels = torch.randint(0, 100, (100,))
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                return self.data[idx], self.labels[idx]
        
        # Test with a dataset that's missing the attributes
        class BadDataset(Dataset):
            def __init__(self):
                # No classes or num_classes attributes!
                self.data = torch.randn(100, 3, 32, 32)
                self.labels = torch.randint(0, 100, (100,))
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                return self.data[idx], self.labels[idx]
        
        # Test with a dataset that has num_classes but not classes
        class MixedDataset(Dataset):
            def __init__(self):
                self.num_classes = 100  # Only num_classes, no classes
                self.data = torch.randn(100, 3, 32, 32)
                self.labels = torch.randint(0, 100, (100,))
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                return self.data[idx], self.labels[idx]
        
        # Test good dataset
        print("=== Testing GoodDataset ===")
        good_dataset = GoodDataset()
        result = simulate_main_py_dataset_check(good_dataset)
        assert result == 100
        
        # Test bad dataset
        print("\n=== Testing BadDataset ===")
        bad_dataset = BadDataset()
        with pytest.raises(AttributeError):
            simulate_main_py_dataset_check(bad_dataset)
        
        # Test mixed dataset
        print("\n=== Testing MixedDataset ===")
        mixed_dataset = MixedDataset()
        result = simulate_main_py_dataset_check(mixed_dataset)
        assert result == 100
    
    def test_what_happens_with_data_loader(self):
        """Test what happens when we wrap datasets in DataLoader"""
        
        class TestDataset(Dataset):
            def __init__(self):
                self.classes = list(range(100))
                self.num_classes = 100
                self.data = torch.randn(100, 3, 32, 32)
                self.labels = torch.randint(0, 100, (100,))
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                return self.data[idx], self.labels[idx]
        
        # Create dataset
        dataset = TestDataset()
        print(f"Original dataset type: {type(dataset)}")
        print(f"Original dataset has classes: {hasattr(dataset, 'classes')}")
        print(f"Original dataset has num_classes: {hasattr(dataset, 'num_classes')}")
        
        # Wrap in DataLoader
        loader = DataLoader(dataset, batch_size=16, shuffle=True)
        print(f"DataLoader dataset type: {type(loader.dataset)}")
        print(f"DataLoader dataset has classes: {hasattr(loader.dataset, 'classes')}")
        print(f"DataLoader dataset has num_classes: {hasattr(loader.dataset, 'num_classes')}")
        
        # The DataLoader should preserve the dataset attributes
        assert hasattr(loader.dataset, 'classes')
        assert hasattr(loader.dataset, 'num_classes')
    
    def test_concat_dataset_behavior(self):
        """Test ConcatDataset behavior"""
        
        class TestDataset(Dataset):
            def __init__(self, classes):
                self.classes = classes
                self.num_classes = len(classes)
                self.data = torch.randn(50, 3, 32, 32)
                self.labels = torch.randint(0, len(classes), (50,))
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                return self.data[idx], self.labels[idx]
        
        # Create two datasets
        dataset1 = TestDataset(list(range(50)))
        dataset2 = TestDataset(list(range(50, 100)))
        
        print(f"Dataset1 has classes: {hasattr(dataset1, 'classes')}")
        print(f"Dataset2 has classes: {hasattr(dataset2, 'classes')}")
        
        # Create ConcatDataset
        concat_dataset = torch.utils.data.ConcatDataset([dataset1, dataset2])
        
        print(f"ConcatDataset type: {type(concat_dataset)}")
        print(f"ConcatDataset has classes: {hasattr(concat_dataset, 'classes')}")
        print(f"ConcatDataset has num_classes: {hasattr(concat_dataset, 'num_classes')}")
        
        # ConcatDataset should NOT have these attributes
        assert not hasattr(concat_dataset, 'classes')
        assert not hasattr(concat_dataset, 'num_classes') 