import pickle

pickle_in = open("data/network_input_500.pkl", "rb") #rb = read bytes
network_input = pickle.load(pickle_in)

print(network_input)
pickle_in.close()

pickle_in2 = open("data/network_output_500.pkl", "rb")
network_output = pickle.load(pickle_in2)

print(network_output)
pickle_in2.close()


# network_output_500.pkl file is not correct. Generate temporary dummy data
import numpy as np
outputs = []
for i in range(500):
    outputs.append(np.asarray([np.random.choice([1,0]) for _ in range(164)]))

#save as pickle file
pickle_out = open("dummy_output.plk", "wb") #wb = write bytes
pickle.dump(outputs, pickle_out)
pickle_out.close()

