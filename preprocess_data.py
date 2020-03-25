import pickle

pickle_in = open("data/network_input_500.pkl", "rb") #rb = read bytes
network_input = pickle.load(pickle_in)

print(network_input)
pickle_in.close()

pickle_in2 = open("data/network_output_500.pkl", "rb")
network_output = pickle.load(pickle_in2)

print(network_output)
pickle_in2.close()