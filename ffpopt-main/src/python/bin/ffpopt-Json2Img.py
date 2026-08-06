#!/usr/bin/env python3


def show_atom_number(mol, label, offset):
    for atom in mol.GetAtoms():
        atom.SetProp(label, str(atom.GetIdx()+1+offset))
    return mol

def show_atom_names(mol, label, names):
    for atom in mol.GetAtoms():
        atom.SetProp(label, names[atom.GetIdx()])
    return mol


def GetSelectedAtomIndices(param,maskstr):
    import parmed
    sele = []
    if len(maskstr) > 0:
        newmaskstr = maskstr.replace("@0","!@*")
        sele = [ param.atoms[i].idx for i in parmed.amber.mask.AmberMask( param, newmaskstr ).Selected() ]
    return sele


class Ligand(object):
    def __init__(self, inp, out, showidxs, shownames, size):
        """ This is a class to generate an image of a ligand with softcore atoms highlighted

        Parameters
        ----------
        inp : str
            Path to json input
        out : str
            Path to png output
        showidxs : bool
            If True, show the atom indexes in the image
        shownames : bool
            If True, show the atom names in the image
        size : tuple of int
            The image size in pixels.

        Returns
        -------
        None

        Raises
        ------
        FileNotFoundError
            If the mdin, parmfile, or rstfile does not exist

        
        """
        from pathlib import Path

        self.resstop=1
        self.inp = Path(inp)
        self.out = Path(out)
        self.showidxs = showidxs
        self.shownames = shownames
        self.size = size
        if not self.inp.exists():
            raise FileNotFoundError(f"{self.mdin} does not exist")
        

    def run(self):
        """ This function runs the class and generates the images """
        self.parm = self._read_parm()
        #self._find_softcore()
        #ti_only, sc_only, ti_all = [], [], []
        #for i in range(len(self.softcore)):
        #    ti_only.append(self._find_difference(self.softcore[i], self.timask[i]))
        #    sc_only.append(self._parse_range(self.softcore[i]))
        #    ti_all.append(self._parse_range(self.timask[i]))

        #self.ti_only = ti_only
        #self.sc_only = sc_only
        #self.ti_all = ti_all
        self._write_2d_structure()
        return

    def _read_parm(self):
        """ This function reads the parameter and restart files and returns a parameter file object"""
        from ffpopt.Struct import ListOfStruct, Struct

        #try:
        los = ListOfStruct.from_file( str(self.inp) )
        #except:
        #    m = ListOfStruct.from_mol2( str(self.inp) )
            
        return los[0].GetParmedAtoms()
        
    
    def _write_2d_structure(self):
        """ This function writes the 2D structure of the ligand with the softcore atoms highlighted 
        
        Parameters
        ----------
        parm : parmed.Structure
            A parmed structure object
            
        Returns
        -------
        None
        
        TODO
        ----
        This function should be modified to check charges on the ligands. 
        
        """
        import rdkit
        from rdkit import Chem
        from rdkit.Chem import Draw, rdDetermineBonds
        from pathlib import Path

        sc = []
        fake_pdb = FakeFile()
        self.parm.write_pdb(fake_pdb)
        if True:
            #for i, mol in enumerate(ligand_resnames):
            mol = self.parm.residues[0].name
            mol_content = fake_pdb.getvalue(mask=[mol])
            rdmol = Chem.MolFromPDBBlock(mol_content, removeHs=False)

            # 1. Gather custom atom labels
            custom_atom_names = [atom.name for atom in self.parm.residues[0].atoms]
            #print("Mapping custom labels:", custom_atom_names)
            
            q1 = round(sum([atom.charge for atom in self.parm.residues[0].atoms]))

            # 2. Rebuild structural bonds (destroys old states)
            rdDetermineBonds.DetermineBonds(rdmol, charge=q1)
            
            Chem.rdDepictor.SetPreferCoordGen(True)
            Chem.rdDepictor.Compute2DCoords(rdmol)

            # 3. Handle structure preparations safely
            Draw.PrepareMolForDrawing(rdmol)

            # if self.shownames:
            #     # 4. Apply text tags at the absolute last moment using universal properties
            #     for atom in rdmol.GetAtoms():
            #         atom_idx = atom.GetIdx()
            #         if atom_idx < len(custom_atom_names):
            #             label_text = str(custom_atom_names[atom_idx])
                    
            #             # _atomLabel overrides the primary drawn symbol
            #             atom.SetProp("_atomLabel", label_text)
            #             # atomNote prints the literal text directly next to the node (Universal fail-safe)
            #             atom.SetProp("atomNote", label_text)


            # # 5. Render directly using default, clean drawing properties
            # img = Draw.rdMolDraw2D.MolDraw2DCairo(*self.size)
            # opts = img.drawOptions()
            # opts.baseFontSize = 0.46

            # img.DrawMolecule(rdmol, highlightAtoms=sc)
            # img.FinishDrawing()

            Path(self.out).resolve().parent.mkdir(parents=True, exist_ok=True)
            
            #with open(str(self.out), 'wb') as f:
            #    f.write(img.GetDrawingText())

            if self.showidxs:
                offset = self.parm.residues[0].atoms[0].idx
                show_atom_number(rdmol,'atomLabel',offset)
            elif self.shownames:
                show_atom_names(rdmol,'atomLabel',custom_atom_names)

            if True:
                img = Draw.rdMolDraw2D.MolDraw2DCairo(*self.size)
                opts = img.drawOptions()
                opts.baseFontSize = 0.42
                opts.useMolBlockWedging = True
                opts.singleColourWedgeBonds = True
                opts.highlightBondWidthMultiplier = 12

                img.DrawMolecule(rdmol,highlightAtoms=sc)
                img.FinishDrawing()

                #idxname = Path(self.out).with_suffix("_idxs.png")
                p = Path(self.out)
                #idxname = p.parent / f"{p.stem}_idxs.png"
                idxname = str(p)
                
                with open(idxname,'wb') as f:
                    f.write(img.GetDrawingText())
            
        return 

class FakeFile(object):
    def __init__(self):
        """ This is a fake file class to capture the output of parmed.write_pdb """
        self.content = ""

    def write(self, text):
        """ This function writes the text to the content 
        
        Parameters
        ----------
        text : str
            The text to write to the content

        """
        self.content += text

    def getvalue(self, mask=None):
        """ This function returns the content of the file 
        
        Parameters
        ----------
        mask : list
            A list of strings to filter the content
            
        Returns
        -------
        str
            The content of the file

        """
        content = []
        if mask == None:
            return self.content
        for line in self.content.split('\n'):
            for elem in mask:
                if elem in line:
                    content.append(line)
        content = '\n'.join(content)
        return content
    
        
if __name__ == "__main__":
    
    import argparse
    from pathlib import Path
    
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="""Reads a json or mol2 file and writes an image of the 2d structure""")
    
    parser.add_argument \
        ("-i","--inp",
         type=str,
         required=True,
         help="json input file")
    
    parser.add_argument \
        ("--out",
         type=str,
         required=True,
         help="The name of the output image file. This should end with .png. The output is {imgdir}/{out}")

    parser.add_argument \
        ("--showidxs",
         action='store_true',
         help="If present, write a second image showing the atom indexes rather than element symbols")
    
    parser.add_argument \
        ("--shownames",
         action='store_true',
         help="If present, write a second image showing the atom names rather than element symbols")
    
    parser.add_argument \
        ("--width",
         type=int,
         default=400,
         help="Width of the image in pixels. Default: 400")
    
    parser.add_argument \
        ("--height",
         type=int,
         default=-1,
         help="Height of the image in pixels. Default: -1, which assumes the same value as --width")


    
    args = parser.parse_args()

    
    if Path(args.out).suffix.lower() != ".png":
        raise Exception(f"--out={args.out} should end in .png")
        
    size = (args.width,args.height)

    ligand = Ligand(args.inp,
                    args.out,
                    args.showidxs,
                    args.shownames,
                    size)

    ligand.run()

    
