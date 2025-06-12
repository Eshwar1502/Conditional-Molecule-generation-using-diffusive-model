from fcd_torch import FCD  # Ensure this package is installed


def compute_intermediate_FCD(smiles, n_jobs=1, device='cpu', batch_size=512):
    """
    Precomputes statistics such as mean and variance for FCD.
    """
    kwargs_fcd = {'n_jobs': n_jobs, 'device': device, 'batch_size': batch_size}
    stats = FCD(**kwargs_fcd).precalc(smiles)
    return stats


def get_FCDMetric(ref_smiles, n_jobs=1, device='cpu', batch_size=512):

    pref = compute_intermediate_FCD(ref_smiles, n_jobs, device, batch_size)

    def FCDMetric(gen_smiles):

        kwargs_fcd = {'n_jobs': n_jobs, 'device': device, 'batch_size': batch_size}
        return FCD(**kwargs_fcd)(gen=gen_smiles, pref=pref)

    return FCDMetric
# Example reference and generated SMILES (replace with your data)
reference_smiles = ['C[C@@H]1CCCN(C(=O)NCc2ccc(N3CCOCC3)cc2)CC1','CC[C@@H](c1cccc(Cl)c1)[C@@H](C)O', 'CC(C)OC(=O)CC(=O)CSC1=C(C=C2CCCC2=N1)C#N']  # Single reference molecule (as a list)

generated_smiles = [
   'CCN1CCN(C)C2=CCC2(O)OC(c2cnc(C)nn2)=NN=N1', 'Cc1csc(C=NNCC(C)S)n1', 'CC1=NC=CC(=Nc2ccccc2Cl)C=C2C(=CC=O)C=C21', 'Cc1cc[n+]2c(n1)C1(CC1)NN2C1CN1', 'COCCC=CNC(=O)N1CC2CC3CC(C=N2)C31', 'COCCCCC=CC=C1C(=O)C2(C(=O)O)CCN1S2(=O)(O)O', 'COCC(C)(CC(Cl)=CC=C1C=CC1=Cc1ccco1)c1cnc(O)[nH]1', 'COC1=CC(O)=CC=CCNC1=O', 'CNC(=O)NCC=CN=C1C2=CC=C1C1=CC(=CC=CC3=CC=CC(=C1)C3)C(=O)N(C)C(=O)N2', 'O=C1CCCC2=C3C=CC=C(C(C=C2)N1)S3(=O)=O', 'CCOC1CCC(O)(CCN2CCCCC2)OC2=CC=C2C(=O)N1', 'CC1C=C2C=C2NC(=O)C(O)C(=O)C(=NO)CN2CC2C(C)O1', 'O=c1c2c(oc(O)ccc3sccn13)=NN=C(NCSO)OCCCC=2', 'CC1CC2=CC(=CC=C3C=C2c2ncc4n(C)c(=O)c-4c3[nH]2)C1F', 'Cc1nc(N(C)CCN2C=NNC3CCCCN3CCNC3=CC=C=CC=CC=C32)no1', 'CCC1C2CN(C(=O)O)C(NC)=NC=Cc3ccccc3N1CCC(CC)N2C', 'OC1=CC=C2C3=CCC1=CC=CC=C[SH]2CC3', 'CN1C(=O)c2cc3ccc2N2CC1=NC2CNC(=O)O3', 'CC(C)Nc1ccc2cc1C=C(C1=CCC3CCC13)CNS2(=O)(O)O', 'CCC[N+]1(CCOC2=CC=C3C=C(CS3(=O)=O)N2)CCN1', 'CCC1=C2NC3CNC(=O)NC(=O)OC4=CCCC(C=CC4=N1)CCC23', 'CN(C1=CC=C2CN(CC1)C(=O)N2)c1cnc[nH]c1=O', 'CC1=CC=CC(C#N)=C(O)C23CC2=CC=CCC=C(C2=CC=C2)C(=O)CS(=O)(=O)C13', 'C[n+]1ncnc2c1N=CC=C(C#N)C2', 'CC1=C2CSC=CC3=NC4=NC5=CC(=CC=C4OC2=N1)C5S(=O)(O)(O)N3', 'CC1CC=CC2=CCN(CCC3CS(=O)C3C2)C1', 'O=S1NCC(F)=CC1=CC=CC=CN1CC1', 'CCC1C2=NC3=CC(=CC=C2O)C[N+](CC)(C3=O)C1C', 'O=C1NCOC=CC=CSC1=Nc1nc(CO)c[nH]1', 'NC(=O)C1=CC2=C(O)C1=CC(CN1CCC3=[N+]=C(C1)S3)=CS2=O', 'COC(C#N)c1cc2ccccc2c2c1=NC=2CCOCCOC(=O)O', 'C[N+](=O)C1=CC2=C3C=C(c4cc(N)c(o4)C=C1F)C2CS(=N)(=O)N3', 'CCN1C=C2CCCCCC=C3CC1C2=CC(=O)OC3=O', 'O=C(O)CC(F)C1=CC=C1C1=CC=C(F)C2=CC=CC=c3oc2ccc2n(c3=CC=C2)C(=O)NC1', 'O=C1C(O)CC(S)CCN=CN1CN1CC1', 'Cc1c(C(F)F)c2n3c1C(=CC=C2)OCC(C)(S)CCN3', 'CCNC1C=CC2=CC=C2C(=O)N2N=C(C)N=Nn3nc1cc32', 'C#CC1NCCN2Oc3cccc(c3)CC1C2=O', 'CNC(=O)CC(=O)c1ccc2c(n1)CO2', 'CCCN1C(=O)SC1=NN1C=C(C)N=CN=C2C=CC=[N+]3C(=O)N2CC=C13', 'CC(=O)Cn1c(S(C)=O)ccc2cc(Cl)c(cc1=O)CO2', 'CCCC1c2ccccc2-c2cc(N3CC3=O)c(=O)n1n2', 'COC1C=CC=CCC(=CC=CF)C=C2CC(=O)C[SH](O)C1(C)O2', 'CC1=CC=C1C1C2=C(CO)C(=Nc3nc1c[nH]3)NC2=O', 'CCCC(=N)C1=CC=CC2=CC=C1NC(=O)N2', 'C1CN(CC2CC[N+]3=NNC3=N2)C1', 'C[N+](=O)C1=CC=C2CCOCC=CC=C2C(=O)NC1', 'O=C(O)Nc1cccc2nc(CCBr)oc12', 'CCN(CCCN=c1c(=O)n1C)CC1(C)CC=CC=N1', 'C#CN=CSC1CCCC2=CC=CC2=CC2=CC=C(Cl)CN=C1N2', 'O=C1NC2=CC=C3NC4=CC=C2C(=O)N3CCC14', 'O=C1C2=CC=C3C=CC(CN4CCNC=C1C4N=C2Cl)C(Br)=CC3', 'CC1CCCN2C(=O)NCC2C(O)=NC1C', 'O=C1C2=NC3=CC=CN=C(NC4=CC=CC=NC(=CC=CC=C(C4)O3)C=C2)OC2CCN1C2', 'CC1=CC2=CC=CC3=CC(=CC=C(C=C3)CC=C1ON=N)C(=O)O2', 'CONC1=CC=C1CS', 'CC1C(=O)N2N=C(N=C(F)C(C)N1C)C2=S', 'O=C1C2=CC3=CC=CC(F)=C4N=C4N=C2SC13', 'Cc1ccc2n(C)cc3c(c1)C(=CC=CC(C)N)N3C2=O', 'CC12CCOC(=C3C=C4N1C(=CN)C1(C)NC(O)(O)C41C3)C2', 'O=C1C2CN1N=c1ncc(cn1)=CC=C1C(Cl)=CC=CC=CC=C1S2', 'CC12CC3=NC=C(O1)C(=NN1C(O)C1C=C3N=CN1CC1)C2', 'CN1C=CC=C2CC=C2C(=O)NC2(O)NN=C2C=CC1', 'CC(C1CC1)N1CCNCCN1C(C=C(C=O)C=O)=C1C=CC=CC=C1O',
'CCN1CCN(C)C2=CCC2(O)OC(c2cnc(C)nn2)=NN=N1', 'Cc1csc(C=NNCC(C)S)n1', 'CC1=NC=CC(=Nc2ccccc2Cl)C=C2C(=CC=O)C=C21', 'Cc1cc[n+]2c(n1)C1(CC1)NN2C1CN1', 'COCCC=CNC(=O)N1CC2CC3CC(C=N2)C31', 'COCCCCC=CC=C1C(=O)C2(C(=O)O)CCN1S2(=O)(O)O', 'COCC(C)(CC(Cl)=CC=C1C=CC1=Cc1ccco1)c1cnc(O)[nH]1', 'COC1=CC(O)=CC=CCNC1=O', 'CNC(=O)NCC=CN=C1C2=CC=C1C1=CC(=CC=CC3=CC=CC(=C1)C3)C(=O)N(C)C(=O)N2', 'O=C1CCCC2=C3C=CC=C(C(C=C2)N1)S3(=O)=O', 'CCOC1CCC(O)(CCN2CCCCC2)OC2=CC=C2C(=O)N1', 'CC1C=C2C=C2NC(=O)C(O)C(=O)C(=NO)CN2CC2C(C)O1', 'O=c1c2c(oc(O)ccc3sccn13)=NN=C(NCSO)OCCCC=2', 'CC1CC2=CC(=CC=C3C=C2c2ncc4n(C)c(=O)c-4c3[nH]2)C1F', 'Cc1nc(N(C)CCN2C=NNC3CCCCN3CCNC3=CC=C=CC=CC=C32)no1', 'CCC1C2CN(C(=O)O)C(NC)=NC=Cc3ccccc3N1CCC(CC)N2C', 'OC1=CC=C2C3=CCC1=CC=CC=C[SH]2CC3', 'CN1C(=O)c2cc3ccc2N2CC1=NC2CNC(=O)O3', 'CC(C)Nc1ccc2cc1C=C(C1=CCC3CCC13)CNS2(=O)(O)O', 'CCC[N+]1(CCOC2=CC=C3C=C(CS3(=O)=O)N2)CCN1', 'CCC1=C2NC3CNC(=O)NC(=O)OC4=CCCC(C=CC4=N1)CCC23', 'CN(C1=CC=C2CN(CC1)C(=O)N2)c1cnc[nH]c1=O', 'CC1=CC=CC(C#N)=C(O)C23CC2=CC=CCC=C(C2=CC=C2)C(=O)CS(=O)(=O)C13', 'C[n+]1ncnc2c1N=CC=C(C#N)C2', 'CC1=C2CSC=CC3=NC4=NC5=CC(=CC=C4OC2=N1)C5S(=O)(O)(O)N3', 'CC1CC=CC2=CCN(CCC3CS(=O)C3C2)C1', 'O=S1NCC(F)=CC1=CC=CC=CN1CC1', 'CCC1C2=NC3=CC(=CC=C2O)C[N+](CC)(C3=O)C1C', 'O=C1NCOC=CC=CSC1=Nc1nc(CO)c[nH]1', 'NC(=O)C1=CC2=C(O)C1=CC(CN1CCC3=[N+]=C(C1)S3)=CS2=O', 'COC(C#N)c1cc2ccccc2c2c1=NC=2CCOCCOC(=O)O', 'C[N+](=O)C1=CC2=C3C=C(c4cc(N)c(o4)C=C1F)C2CS(=N)(=O)N3', 'CCN1C=C2CCCCCC=C3CC1C2=CC(=O)OC3=O', 'O=C(O)CC(F)C1=CC=C1C1=CC=C(F)C2=CC=CC=c3oc2ccc2n(c3=CC=C2)C(=O)NC1', 'O=C1C(O)CC(S)CCN=CN1CN1CC1', 'Cc1c(C(F)F)c2n3c1C(=CC=C2)OCC(C)(S)CCN3', 'CCNC1C=CC2=CC=C2C(=O)N2N=C(C)N=Nn3nc1cc32', 'C#CC1NCCN2Oc3cccc(c3)CC1C2=O', 'CNC(=O)CC(=O)c1ccc2c(n1)CO2', 'CCCN1C(=O)SC1=NN1C=C(C)N=CN=C2C=CC=[N+]3C(=O)N2CC=C13', 'CC(=O)Cn1c(S(C)=O)ccc2cc(Cl)c(cc1=O)CO2', 'CCCC1c2ccccc2-c2cc(N3CC3=O)c(=O)n1n2', 'COC1C=CC=CCC(=CC=CF)C=C2CC(=O)C[SH](O)C1(C)O2', 'CC1=CC=C1C1C2=C(CO)C(=Nc3nc1c[nH]3)NC2=O', 'CCCC(=N)C1=CC=CC2=CC=C1NC(=O)N2', 'C1CN(CC2CC[N+]3=NNC3=N2)C1', 'C[N+](=O)C1=CC=C2CCOCC=CC=C2C(=O)NC1', 'O=C(O)Nc1cccc2nc(CCBr)oc12', 'CCN(CCCN=c1c(=O)n1C)CC1(C)CC=CC=N1', 'C#CN=CSC1CCCC2=CC=CC2=CC2=CC=C(Cl)CN=C1N2', 'O=C1NC2=CC=C3NC4=CC=C2C(=O)N3CCC14', 'O=C1C2=CC=C3C=CC(CN4CCNC=C1C4N=C2Cl)C(Br)=CC3', 'CC1CCCN2C(=O)NCC2C(O)=NC1C', 'O=C1C2=NC3=CC=CN=C(NC4=CC=CC=NC(=CC=CC=C(C4)O3)C=C2)OC2CCN1C2', 'CC1=CC2=CC=CC3=CC(=CC=C(C=C3)CC=C1ON=N)C(=O)O2', 'CONC1=CC=C1CS', 'CC1C(=O)N2N=C(N=C(F)C(C)N1C)C2=S', 'O=C1C2=CC3=CC=CC(F)=C4N=C4N=C2SC13', 'Cc1ccc2n(C)cc3c(c1)C(=CC=CC(C)N)N3C2=O', 'CC12CCOC(=C3C=C4N1C(=CN)C1(C)NC(O)(O)C41C3)C2', 'O=C1C2CN1N=c1ncc(cn1)=CC=C1C(Cl)=CC=CC=CC=C1S2', 'CC12CC3=NC=C(O1)C(=NN1C(O)C1C=C3N=CN1CC1)C2', 'CN1C=CC=C2CC=C2C(=O)NC2(O)NN=C2C=CC1', 'CC(C1CC1)N1CCNCCN1C(C=C(C=O)C=O)=C1C=CC=CC=C1O'
]     # Generated molecules

# Step 1: Create the FCD metric using the reference SMILES
fcd_metric = get_FCDMetric(ref_smiles=reference_smiles, device='cpu')  # Use 'cuda' for GPU

# Step 2: Compute FCD between reference and generated SMILES
fcd_distance = fcd_metric(gen_smiles=generated_smiles)

print(f"FCD Distance: {fcd_distance}")