import pickle
# Load from the file
with open("CO2_Hamiltonian.pkl", "rb") as f:
    H_loaded = pickle.load(f)

print("Hamiltonian loaded and ready for VQE.")

print(H_loaded)