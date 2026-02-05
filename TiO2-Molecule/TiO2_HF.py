from pyscf import gto, scf

mol = gto.M(
    atom = '''
        Ti   0.000   0.000   0.000;
        O    1.620   0.000   0.000;
        O   -0.810   0.000   1.403
    ''',
    # basis = 'def2-svp' is recommended for Titanium (better than 6-31G) 
    basis = '6-31G',
    spin = 0,  # Spin = 2*S. For a Singlet (S=0), we know spin = 0
    charge = 0, # neutral TiO
    symmetry = True # if symmetry is on, calculation is faster
)

mol.build()
print(f"Number of electrons: {mol.nelectron}")
print(f"Spin multiplicity: {mol.spin + 1}")
mf = scf.RHF(mol) # Hartree Fock calculation (closed-shell system)
hf_energy = mf.kernel()
print(f"Total HF Energy: {hf_energy} Hartree")
