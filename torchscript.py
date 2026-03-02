import torch
import pickle
from util import testdata_generate # Ensure your model/dataset logic is available

# 1. Load model as you did in runmodel.py
with open('save/cnnlstm_e5_b8ed7038fcc611f09084bab6975ba98b.pkl', 'rb') as f:
    model = pickle.load(f)
model.eval()
model.cpu() # Trace on CPU for better C++ compatibility initially

# 2. Get a REAL sample from your loader
path_validate = "test_split/FathersDay/FathersDayR0P0Y0_0"
test_dataset = testdata_generate.VideoFrameDataset(path_validate, 1)
test_loader = torch.utils.data.DataLoader(test_dataset)

# Extract one batch
inputs, targets, masks, max_seq, position_intervals = next(iter(test_loader))

# 3. Trace with the actual data shapes
# We pass (inputs, masks) as a tuple because your model takes two arguments
try:
    traced_model = torch.jit.trace(model, (inputs, masks))
    traced_model.save("model_trace.pt")
    print("Success! Model saved as model_trace.pt")
except Exception as e:
    print(f"Tracing failed: {e}")
    print(f"Input shape used: {inputs.shape}") # Should be [1, Seq, 3, 224, 224]