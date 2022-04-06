#!/usr/bin/env python3

import itertools
import logging
from datetime import datetime
from multiprocessing import Manager
from pathlib import Path
from typing import Dict, List, Union, Tuple, Iterable

import freesasa
import numpy as np
import rmsd
from Bio.PDB import NeighborSearch
from Bio.PDB.Atom import Atom
from Bio.PDB.Chain import Chain
from Bio.PDB.Entity import Entity
from Bio.PDB.Residue import Residue

from analyses import configure_pipeline

from apo_holo_structure_stats.pipeline.utils.log import add_loglevel_args
from apo_holo_structure_stats import project_logger
from apo_holo_structure_stats.core.analyses import get_defined_ligands
from apo_holo_structure_stats.core.biopython_to_mmcif import BiopythonToMmcifResidueIds, BioResidueId
from apo_holo_structure_stats.core.dataclasses import SetOfResidues
from apo_holo_structure_stats.pipeline.make_pairs_lcs import LCSResult
from apo_holo_structure_stats.pipeline.run_analyses import ConcurrentJSONAnalysisSerializer, plocal, \
    assert_label_seq_id_contiguous, run_analyses_serial, run_analyses_multiprocess
from apo_holo_structure_stats.pipeline.run_analyses_settings import NotXrayDiffraction

from run_analyses_settings import BLOCKING_DISTANCE, BS_RADIUS

logger = logging.getLogger()




# get atoms a access O(1) by id
class SetOfResiduesIndexable(SetOfResidues):
    """ Residues can be indexed by integer identifier. """
    def __init__(self, residues: Dict[int, Residue], s_pdb_code):
        super().__init__(residues, s_pdb_code)

    def __iter__(self):
        return iter(self.data.values())

    @property
    def label_seq_ids(self):
        return self.data.keys()

    @classmethod
    def from_label_seq_ids(cls, label_seq_ids: Iterable[int], mapping: BiopythonToMmcifResidueIds.Mapping,
                           bio_chain: Chain):
        return cls(
            {ls_id: bio_chain[mapping.to_bio(ls_id)] for ls_id in label_seq_ids},
            bio_chain.get_parent().get_parent().id,
        )

    @classmethod
    def from_residues(cls, residues: Iterable[Residue], mapping: BiopythonToMmcifResidueIds.Mapping, structure_pdb_code: str):
        return cls(
            {mapping.to_label_seq_id(BioResidueId.from_bio(r)): r for r in residues},
            structure_pdb_code,
        )


def get_atoms_near_other_atoms(subject: Entity, object_atoms: List[Atom], radius: float, return_contacts=False) \
        -> Union[List[Atom], List[Tuple[Atom, Atom]]]:
    # had this method written using different terms
    chain = subject
    ligand_atoms = object_atoms

    chain_atoms = list(chain.get_atoms())
    ns = NeighborSearch(chain_atoms)

    atom_contacts = []
    atoms_in_contact_with_ligand = set()  # including the ligand itself (in biopython, non-peptide ligand is
    # in the same chain usually, but in a different residue)

    ligand_residues = set()  # residues that compose the ligand
    ligand_atoms = set(ligand_atoms)

    for ligand_atom in ligand_atoms:  # ligand can be a chain or a residue
        ligand_residues.add(ligand_atom.get_parent())
        chain_atoms_in_contact = ns.search(ligand_atom.get_coord(), radius)

        for atom in chain_atoms_in_contact:
            # exclude hydrogen atoms
            if atom.element == 'H':
                continue

            # exclude the ligand itself from the set of contact atoms
            if atom in ligand_atoms:
                continue

            atoms_in_contact_with_ligand.add(atom)
            if return_contacts:
                atom_contacts.append((atom, ligand_atom))

    if return_contacts:
        return atoms_in_contact_with_ligand, atom_contacts

    return atoms_in_contact_with_ligand


def get_binding_site_atoms(chain: Chain, ligand: Entity, radius: float):
    return get_atoms_near_other_atoms(chain, ligand.get_atoms(), radius)


def remap_atoms_to_other_chain(chain1_atoms, chain2, c1_mapping: BiopythonToMmcifResidueIds.Mapping, c2_mapping, c2_label_seq_id_offset, insert_none=False):
    chain2_atoms = []
    chain1_atoms_also_in_chain2 = []  # to have a one-to-one mapping

    for c1_atom in chain1_atoms:
        ls_id = c1_mapping.to_label_seq_id(BioResidueId.from_bio(c1_atom.get_parent()))

        residue_or_atom_unobserved_in_chain2 = False

        try:
            c2_r = chain2[ls_id + c2_label_seq_id_offset]
        except KeyError:
            # unobserved residue (or completely missing from the experiment protein sequence) in chain2
            # todo we could could chain1_atoms as blocking, if the aren't in the holo protein experimental sequence,
            #   but I would need add a similar method to this one and maybe also change other things downstream
            #   (this way we omit some cryptic binding sites, but they would be cryptic due to being a different protein,
            #     so in a sense that's ok)
            logger.warning('apo residue unobserved or missing completely in holo structure')
            residue_or_atom_unobserved_in_chain2 = True

        if not residue_or_atom_unobserved_in_chain2:
            try:
                c2_atom = c2_r[c1_atom.get_id()]  # get the same atom (id is basically its name, e.g. 'CA') in the corresponding residue
                chain2_atoms.append(c2_atom)
                chain1_atoms_also_in_chain2.append(c1_atom)  # to have a one-to-one mapping
            except KeyError:
                # unobserved BS atom in chain2 (that is observed in chain1)
                residue_or_atom_unobserved_in_chain2 = True

        # one might want to keep the shape
        if residue_or_atom_unobserved_in_chain2 and insert_none:
            chain2_atoms.append(None)
            chain1_atoms_also_in_chain2.append(None)

    return chain2_atoms, chain1_atoms_also_in_chain2


def serialize_ligand(ligand):
    if isinstance(ligand, Chain):
        # peptide ligand
        return '-'.join(r.resname for r in ligand)
    elif isinstance(ligand, Residue):
        # residue of HETATMs
        return ligand.resname

    raise NotImplementedError


def get_seqs_range_and_offset(seq1: Dict[int, str], seq2: Dict[int, str], lcs_result: LCSResult):
    # sanity check I forgot to do previously
    for seq in (seq1, seq2):
        try:
            assert_label_seq_id_contiguous(list(seq.keys()))
        except AssertionError:
            logger.warning('ERROR, label_seq list is not contiguous as expected. What now?')
            try:
                assert_label_seq_id_contiguous(sorted(seq.keys()))
            except AssertionError:
                logger.warning('ERROR, even if sorted, label_seq list is not contiguous as expected. What now?')

    def get_seq_range(seq, lcs_start):
        seq_ids = list(seq.keys())
        start = seq_ids[lcs_start]
        return range(start, start + lcs_result.length + 1)

    seq1_range = get_seq_range(seq1, lcs_result.i1)
    seq2_range = get_seq_range(seq2, lcs_result.i2)
    seq2_label_seq_id_offset = seq2_range[0] - seq1_range[0]
    return seq1_range, seq2_range, seq2_label_seq_id_offset


def process_pair(s1_pdb_code: str, s2_pdb_code: str, s1_chain_code: str, s2_chain_code: str,
                 lcs_result: LCSResult):
    pair = ((s1_pdb_code, s1_chain_code), (s2_pdb_code, s2_chain_code))
    logger.info(f'process_pair {pair}')

    try:
        apo_parsed = plocal.parse_mmcif(s1_pdb_code)
        holo_parsed = plocal.parse_mmcif(s2_pdb_code)
    except NotXrayDiffraction:
        logger.exception('not x-ray diffraction')
        logger.info(f'skipping pair {(s1_pdb_code, s2_pdb_code)}: exp. method '
                    f'is not X-RAY DIFFRACTION for a structure of the pair')
        return

    # goes to serialization
    result_dict = {
        'apo_pdb_code': s1_pdb_code,
        'holo_pdb_code': s2_pdb_code,
        'apo_chain_id': s1_chain_code,
        'holo_chain_id': s2_chain_code,
        'binding_sites': [],
    }

    apo = apo_parsed.structure
    apo_residue_id_mappings = apo_parsed.bio_to_mmcif_mappings
    apo_poly_seqs = apo_parsed.poly_seqs

    holo = holo_parsed.structure
    holo_residue_id_mappings = holo_parsed.bio_to_mmcif_mappings
    holo_poly_seqs = holo_parsed.poly_seqs

    # get the first model (s[0]), (in x-ray structures there is probably a single model)
    apo, apo_residue_id_mappings = map(lambda s: s[0], (apo, apo_residue_id_mappings))
    holo, holo_residue_id_mappings = map(lambda s: s[0], (holo, holo_residue_id_mappings))

    apo_chain = apo[s1_chain_code]
    holo_chain = holo[s2_chain_code]

    apo_mapping: BiopythonToMmcifResidueIds.Mapping = apo_residue_id_mappings[apo_chain.id]
    holo_mapping: BiopythonToMmcifResidueIds.Mapping = holo_residue_id_mappings[holo_chain.id]

    apo_seq = apo_poly_seqs[apo_mapping.entity_poly_id]
    holo_seq = holo_poly_seqs[holo_mapping.entity_poly_id]

    # from now on, the (longest) common sequence:
    apo_seq_range, holo_seq_range, holo_label_seq_id_offset = get_seqs_range_and_offset(apo_seq, holo_seq, lcs_result)
    # the sequence may contain unobserved residues

    # todo
    #  - [x] ted staci implementovat jen ty zpusoby...
    #  - [x] serializace (uz vim, jak ma vypadat example output, snad pujde nejak dat do dataframu)
    #  - pak otestovat lokalne na 100 strukturach
    #  - pak udelat metacentrum skript (zkopirovat ten jiny? Nebo jen pridat parametr mozna staci.., zkopirovat - rychlejsiú
    #       - nakonec spustit (ale nektery neprobehnou kvuli pretizeni pipu)

    # get the ligand(s)
    ligands = list(get_defined_ligands(holo, holo_chain))
    assert len(ligands) > 0  # otherwise wouldn't be holo (as I leave the same LigandSpec)

    # # up to this point, we have residue ids of the protein sequence in the experiment. This also includes unobserved
    # # residues, but those we will exclude from our analysis as their positions weren't determined
    # c1_residues, c1_label_seq_ids, c2_residues, c2_label_seq_ids = get_observed_residues(
    #     apo_chain,
    #     list(apo_seq_range),
    #     apo_mapping,
    #     holo_chain,
    #     list(holo_seq_range),
    #     holo_mapping,
    # )

    # all amino acid sequence residues (won't include heteroatoms)
    # potentially larger than then LCS, and residues not necessarily observed in both structures
    apo_chain = SetOfResiduesIndexable.from_label_seq_ids(apo_mapping.label_seq_id__to__bio_pdb.keys(), apo_mapping,
                                                             apo_chain)
    # asi pridelej label seq id do chain residues?

    holo_chain = SetOfResiduesIndexable.from_label_seq_ids(holo_mapping.label_seq_id__to__bio_pdb.keys(), holo_mapping,
                                                             holo_chain)

    for ligand in ligands:
        # todo serialize the ligand?
        # nase ligandy jsou ale maly/rozdeleny oproti tomu (mozna by mely byt vsechny v jednom chainu, ted nevim, nekdy
        # jsou integrovany v tom branching information? Ne, to se musi vic prozkoumat. Napriklad u 1bs1 paper reportuje
        # DAA;ADP-DAA; pritom tam je jedno DAA jedno ADP a kazdy v jinym chainu)
        # aa codes if polypeptide, residue name if heteroatom = one heteroresidue hopefully.

        # for aa chain '-' joined resnames
        # for hetatm residue the resname



        # get chain atoms in contact with the ligand (the binding site)
        binding_atoms__holo = get_binding_site_atoms(holo_chain, ligand, BS_RADIUS)

        # the binding_atoms' residues.. (are observed, follows from the previous line)
        binding_residues__holo = {atom.get_parent() for atom in binding_atoms__holo}
        binding_residues__holo = SetOfResiduesIndexable.from_residues(binding_residues__holo, holo_mapping, s2_pdb_code)

        # todo, pokud nebude residue observed v apo, tak to tady spadne, ale to nevadí, protože by se to pak stejně
        #   nespolehlivě porovnávalo
        binding_residue_ids__apo = {label_seq_id - holo_label_seq_id_offset for label_seq_id in binding_residues__holo.label_seq_ids}
        binding_residues__apo = SetOfResiduesIndexable.from_label_seq_ids(binding_residue_ids__apo, apo_mapping, apo[s1_chain_code])

        # binding site should be present in its entirety in both structures, so check that first
        # (if not present in LCS skip this ligand)
        if not set(apo_seq_range).issuperset(binding_residue_ids__apo):
            # (same for the holo follows (due to the LCS))
            logger.info(f'skipping binding site for pair {pair}: binding site not in LCS')
            continue

        result_binding_site = {
            'ligand': serialize_ligand(ligand),
            'residue_ids': list(binding_residues__holo.label_seq_ids),  # the holo label seq ids of course
            'analyses': {}
        }
        result_dict['binding_sites'].append(result_binding_site)

        # remap the binding site atoms to the apo structure
        binding_atoms__apo, binding_atoms__holo_also_in_apo = remap_atoms_to_other_chain(binding_atoms__holo, apo_chain,
                                                                                         holo_mapping, apo_mapping,
                                                                                         - holo_label_seq_id_offset)

        # todo ve vyjimecnem pripade muzou byt prazdne oba listy

        # unobserved atoms in BS apo, holo (BS=residues)
        # nojo, těžko říct, to bych musel znát počty atomů pro každý residue asi, ne?
        # a pokud by vždycky byly unobserved ty stejný, tak to je celkem ok, ne?
        #  intersection, diff jeden, diff druhej (diff celkovej ani nezjistim rychle urcite ne)
        # otazka, jestli nas zajimaji binding atoms nebo vsechny binding residue atoms, no zejo, takze 5 hodnot

        def zip_residues(r1: SetOfResiduesIndexable, r2: SetOfResiduesIndexable, label_seq_offset):
            assert len(r1) == len(r2)
            for ls_id in r1.label_seq_ids:
                yield r1[ls_id], r2[ls_id + label_seq_offset]

        # observed atoms count
        count_binding_atoms_in_holo = len(binding_atoms__holo)
        count_binding_atoms_in_apo = len(binding_atoms__apo)

        binding_residues__zipped = list(zip_residues(binding_residues__apo, binding_residues__holo, holo_label_seq_id_offset))
        count_binding_residues_atoms_intersection = sum(len(set(a.id for a in apo_r.get_atoms())
                                                           & set(h.id for h in holo_r.get_atoms()))
                                                       for apo_r, holo_r in binding_residues__zipped)

        binding_residues_atoms_holo_complement = itertools.chain.from_iterable((set(a.id for a in apo_r.get_atoms())
                                                           - set(h.id for h in holo_r.get_atoms()))
                                                       for apo_r, holo_r in binding_residues__zipped)

        binding_residues_atoms_apo_complement = itertools.chain.from_iterable((set(h.id for h in holo_r.get_atoms())
                                                           - set(a.id for a in apo_r.get_atoms()))
                                                       for apo_r, holo_r in binding_residues__zipped)

        binding_residues_atoms_holo_complement = list(binding_residues_atoms_holo_complement)
        binding_residues_atoms_apo_complement = list(binding_residues_atoms_apo_complement)

        result_binding_site['atom_count_statistic'] = {
            'binding_atoms': count_binding_atoms_in_holo,
            'binding_atoms_observed_in_apo': count_binding_atoms_in_apo,
            'atoms_of_binding_residues_in_both': count_binding_residues_atoms_intersection,
            'atoms_of_binding_residues_missing_in_holo':  len(binding_residues_atoms_holo_complement),
            'atoms_of_binding_residues_missing_in_apo':  len(binding_residues_atoms_apo_complement),
        }

        # binding site should be present in its entirety in both structures, so check that first...
        # and obtain mapping between the atoms (not all need to be observed... I only checked for c_alphas I think)
        # if some are not observed, the SASA will be different! So take that into account (have it as another output,
        # so we can report the confidence)
        # however, rmsd, new contacts (residue therefore!, or both atoms and residues I can have) etc will still work
        # but I won't know for sure (unobserved atoms outside of active site)
        #     - however, I can keep the statistics of unobserved atoms in the active site, and, in case of the opening,
        #      in the interacting site (name it that way, active site + the atoms interacting with it in apo form)
        #           - so these residues mapped to both structures


        # compute analyses

        # METHOD 1
        # RMSD of the binding site (heavy atoms) (c_alpha atoms) (all carbon atoms)
        def get_rmsd(atoms1: List[Atom], atoms2: List[Atom]):
            return rmsd.kabsch_rmsd(np.array([a.get_coord() for a in atoms1]),
                                    np.array([a.get_coord() for a in atoms2]), translate=True)

        # heavy atoms
        bs_heavy_atoms__apo = list(filter(lambda a: a.element != 'H', binding_atoms__apo))
        bs_heavy_atoms__holo = list(filter(lambda a: a.element != 'H', binding_atoms__holo_also_in_apo))
        heavy_rmsd = get_rmsd(bs_heavy_atoms__apo, bs_heavy_atoms__holo)

        # carbon atoms
        bs_carbon_atoms__apo = list(filter(lambda a: a.element == 'C', binding_atoms__apo))
        bs_carbon_atoms__holo = list(filter(lambda a: a.element == 'C', binding_atoms__holo_also_in_apo))
        carbon_rmsd = get_rmsd(bs_carbon_atoms__apo, bs_carbon_atoms__holo)

        # c_alpha atoms
        bs_c_alpha_atoms__apo = list(filter(lambda a: a.id == 'CA', binding_atoms__apo))
        bs_c_alpha_atoms__holo = list(filter(lambda a: a.id == 'CA', binding_atoms__holo_also_in_apo))
        c_alpha_rmsd = get_rmsd(bs_carbon_atoms__apo, bs_carbon_atoms__holo)

        result_binding_site['analyses']['method1'] = {
            'heavy_rmsd': heavy_rmsd,
            'carbon_rmsd': carbon_rmsd,
            'c_alpha_rmsd': c_alpha_rmsd,
        }

        # METHOD 2
        # (domains of the binding site and their movements? Movements of the binding site parts?)
        # partition binding site residues by the domains, measure relative centroid translation of the partitions
        #  - (for each pair of partitions), report maximum or the whole list (bad for memory when pd.loaded)
        # (and possibly add the _whole domain_ rotation angle, but that's not that important in my opinion)
        # todo

        # METHOD 3
        # SASA of the binding site (put the whole chain (without ligand), compute SASA, extract only BS atoms' SASA and
        # compare values for apo and holo)

        # or maybe create freesasa structure with defined atom order (but how to set-ify atoms? and keep order?)
        # not that hard
        # but the I have to build the structure with addAtom manually
        def build_freesasa_structure(atoms: List[Atom]):
            s = freesasa.Structure()
            for a in atoms:
                r = a.get_parent()
                hetflag, resseq, icode = r.get_id()
                s.addAtom(a.name, r.resname, str(resseq) + icode, '_', *a.get_coord())  # '_' for 1letter chain id as we don't need it
            return s

        # have the binding atoms as the first, so we know their indices
        apo_sasa_structure = build_freesasa_structure(
            binding_atoms__apo + list(set(apo_chain.get_atoms(exclude_H=True)) - set(binding_atoms__apo)))
        holo_sasa_structure = build_freesasa_structure(
            binding_atoms__holo_also_in_apo + list(set(holo_chain.get_atoms(exclude_H=True)) - set(binding_atoms__holo_also_in_apo)))

        apo_sasa_result = freesasa.calc(apo_sasa_structure)
        holo_sasa_result = freesasa.calc(holo_sasa_structure)

        assert len(binding_atoms__apo) == len(binding_atoms__holo_also_in_apo)
        bs_atoms_sasa__apo = sum(apo_sasa_result.atomArea(i) for i in range(len(binding_atoms__apo)))
        bs_atoms_sasa__holo = sum(holo_sasa_result.atomArea(i) for i in range(len(binding_atoms__apo)))

        result_binding_site['analyses']['method3'] = {
            'bs_atoms_sasa__apo': bs_atoms_sasa__apo,
            'bs_atoms_sasa__holo': bs_atoms_sasa__holo,
        }

        # METHOD 4
        # new contacts on binding site atoms (threshold distance for definition of contact)
        # atom contacts or residue?  Both statistics (new atom contacts might be from a BS residue theoretically)

        # todo not only apo_chain (which is only the LCS!), but also all available would be better
        method_4_result = get_new_contacts_in_apo_bs(apo_chain, binding_atoms__apo, holo_chain, binding_atoms__holo_also_in_apo,
                                                 apo_mapping, holo_mapping, holo_label_seq_id_offset, binding_residues__holo)

        result_binding_site['analyses']['method4'] = method_4_result

        # METHOD 5
        # ((ligand superposition to the BS in apo, steric clashes?))
        # steric clashes is not hard - a measure of that could be the sum of 'negative distances' of contacts between
        # the ligand and the chain's atoms.

        # however, this method is sensitive on the changes of the binding site configuration
        # (say apo=closed, this would report the clashes, but if apo=open, this, depending on the protein structure,
        # could detect some clashes too -- molecular dynamics would fix this.) When a person imagines that, they
        # try to fit the ligand there too, as if doing the simulation. But
        # todo

        # (some methods work for the closing (1,2), others for the opening ([1],[2], 3, 4, 5, none for both probably)

    # serializace: zrusit to tuplovani argumentu, unfold to columns atp.
    # bude to zrat i mene memory (teda asi kdyz to neulozim jako records, coz nemusim, ted kdyz to znam...)
    plocal.serializer.queue.put(result_dict)


def get_new_contacts_in_apo_bs(apo_chain, binding_atoms__apo, holo_chain, binding_atoms__holo_also_in_apo,
                               apo_mapping, holo_mapping, holo_label_seq_id_offset, binding_residues__holo):
    apo_potentially_blocking_atoms, apo_potentially_blocking_contacts = get_atoms_near_other_atoms(
        apo_chain, binding_atoms__apo, BLOCKING_DISTANCE, return_contacts=True)

    holo_totally_not_blocking_atoms, holo_totally_not_blocking_contacts = get_atoms_near_other_atoms(
        holo_chain, binding_atoms__holo_also_in_apo, BLOCKING_DISTANCE, return_contacts=True)

    # blocking_atoms domain: chain\bs(only atoms)
    # contacts domain: tuples of atoms of (chain\bs, bs)

    # really, I should check the new contacts, not just the new atoms?
    # I can do this, filter the contacts

    # do a difference probably_blocking_atoms = apo_potentially_blocking_atoms - holo_totally_not_blocking_atoms
    # i.e. those new contacts in apo, that aren't observed in holo structure
    # could /should/ also filter it so that the probably_blocking are observed in _both_ structures

    # but how to do the set difference in different structures?
    # remap the atoms to the same structures (also that way you get the common ones), however, need to do get the common
    # one for both structures, (therefore remap one twice?)

    # todo
    # no, if holo_totally_not_blocking_atoms contains some extra, the result stays the same, so just ensure that all
    # apo_potentially_blocking_atoms are in the holo structure, and that will be done by the remapping

    potentially_blocking_atoms__remapped_to_holo, _ = remap_atoms_to_other_chain(
        apo_potentially_blocking_atoms, holo_chain,
        apo_mapping, holo_mapping,
        holo_label_seq_id_offset
    )

    blocking_atoms = set(potentially_blocking_atoms__remapped_to_holo) - set(holo_totally_not_blocking_atoms)
    blocking_residues = set(a.get_parent() for a in blocking_atoms)

    # ale chci asi znát i kontakty, abych věděl, jaký procento BS je bloklý.. takže:
    #   "apo_potentially_blocking_contacts - holo_totally_not_blocking_contacts"

    potentially_blocking_contacts__remapped_to_holo = np.array(remap_atoms_to_other_chain(
        np.array(apo_potentially_blocking_contacts).reshape([-1]), holo_chain,
        apo_mapping, holo_mapping,
        holo_label_seq_id_offset, insert_none=True
    )[0]).reshape([-1, 2])

    # convert back to list of tuples (hashable)
    potentially_blocking_contacts__remapped_to_holo = list(map(tuple, potentially_blocking_contacts__remapped_to_holo))

    # remove the contacts with blocking atoms that are not observed in holo
    potentially_blocking_contacts__remapped_to_holo = list(filter(lambda c: c[0] is not None,
                                                             potentially_blocking_contacts__remapped_to_holo))

    blocking_contacts = set(potentially_blocking_contacts__remapped_to_holo) - set(holo_totally_not_blocking_contacts)
    # blocking contacts  `(blocking_atom, binding_site_atom)`, (only for the blocking_atoms from both structures)
    # blocked BS atoms  == set(c[1] for c in blocking_contacts)
    # calling a (private) method that could be a public static method
    mock_self = object()
    blocking_contacts__residue = NeighborSearch._get_unique_parent_pairs(mock_self, blocking_contacts)

    blocked_bs_atoms = set(c[1] for c in blocking_contacts)
    blocked_bs_residues = set(a.get_parent() for a in blocked_bs_atoms)

    # more strict definition of blocking contact: (not counting those with blocker in the BS, in the sense of residues,
    # not just the binding atoms)
    all_binding_site_atoms = {a for r in binding_residues__holo for a in r.get_atoms()}

    blocking_contacts__outside_bs = list(filter(lambda c: c[0] not in all_binding_site_atoms, blocking_contacts))
    blocking_contacts__outside_bs__residue = NeighborSearch._get_unique_parent_pairs(mock_self, blocking_contacts__outside_bs)

    blocked_bs_atoms__by_outside_bs = set(c[1] for c in blocking_contacts__outside_bs)
    blocked_bs_residues__by_outside_bs = set(a.get_parent() for a in blocked_bs_atoms__by_outside_bs)

    # most strict definition of blocking contact (by atoms that were not present in vicinity of BS in the holo structure
    # what so ever
    blocking_contacts__by_blocking_atoms = list(filter(lambda c: c[0] in blocking_atoms, blocking_contacts))
    blocking_contacts__by_blocking_atoms__residue = NeighborSearch._get_unique_parent_pairs(mock_self, blocking_contacts__by_blocking_atoms)

    blocked_bs_atoms__by_blocking_atoms_bs = set(c[1] for c in blocking_contacts__by_blocking_atoms)
    blocked_bs_residues__by_blocking_atoms_bs = set(a.get_parent() for a in blocked_bs_atoms__by_blocking_atoms_bs)


    # only chain level, ignore existence of other chains (everywhere here, except in ligand detection)
    # could also employ other chains in: the sasa method (3) and here (4) and (5), and also (1) and (2), by the extension
    # of binding site definition to allow multi-chain BS.
    result = {
        'new_apo_bs_contacts__atom': len(blocking_contacts),
        'new_apo_bs_contacts__residue': len(blocking_contacts__residue),
        'apo_bs_blocked_atoms': len(blocked_bs_atoms),
        'apo_bs_blocked_residues': len(blocked_bs_residues),

        'apo_bs_blocking_atoms': len(blocking_atoms),
        'apo_bs_blocking_residues': len(blocking_residues),

        'new_apo_bs_contacts__outside_bs__atom': len(blocking_contacts__outside_bs),
        'new_apo_bs_contacts__outside_bs__residue': len(blocking_contacts__outside_bs__residue),
        'apo_bs_blocked_atoms__by_outside_bs': len(blocked_bs_atoms__by_outside_bs),
        'apo_bs_blocked_residues__by_outside_bs': len(blocked_bs_residues__by_outside_bs),

        'new_apo_bs_contacts__by_blocking_atoms__atom': len(blocking_contacts__by_blocking_atoms),
        'new_apo_bs_contacts__by_blocking_atoms__residue': len(blocking_contacts__by_blocking_atoms__residue),
        'apo_bs_blocked_atoms__by_blocking_atoms': len(blocked_bs_atoms__by_blocking_atoms_bs),
        'apo_bs_blocked_residues__by_blocking_atoms': len(blocked_bs_residues__by_blocking_atoms_bs),
        # 'new_apo_bs_contacts__domains':
    }
    return result

def worker_initializer(analyzer_namespace, serializer): #, domains_info, one_struct_analyses_done_set):
    attrs = locals()
    del attrs['analyzer_namespace']
    attrs.update(analyzer_namespace)

    for attr_name, value in attrs.items():
        setattr(plocal, attr_name, value)


def process_pair_with_exc_handling(*args):
    try:
        process_pair(*args)
    except Exception as e:
        logger.exception('process pair failed with: ')


def main():
    # runs for all isoforms by default
    # optionally specify a single isoform with --isoform
    import argparse

    parser = argparse.ArgumentParser()
    # parser.add_argument('--limit_pairs_for_group', type=int, help='process only structures with main chain of that isoform')
    parser.add_argument('--workers', default=4, type=int, help='process only structures with main chain of that isoform')
    parser.add_argument('--opt_input_dir', type=Path, default=Path())
    # parser.add_argument('chains_json', help='list of structures {pdb_code: , path: , isoform_id: , is_holo: bool, ?main_chain_id: }')
    parser.add_argument('pairs_json', help='list of structures {pdb_code: , path: , isoform_id: , is_holo: bool, ?main_chain_id: }')
    add_loglevel_args(parser)

    args = parser.parse_args()
    project_logger.setLevel(args.loglevel)
    logger.setLevel(args.loglevel)  # bohužel musim specifikovat i tohle, protoze takhle to s __name__ funguje...
    logging.basicConfig()

    # don't run analyses for each isoform group separately, as creating a process pool carries an overhead
    # median pairs per group is 6
    # but could reset the caches? No need, all are LRU..
    start_datetime = datetime.now()
    analyses_output_fpath = Path(f'output_apo_holo_{start_datetime.isoformat()}.json')

    with Manager() as multiprocessing_manager:
        # get analyzers as configured
        # p = configure_pipeline(multiprocessing_manager)
        analyses_namespace = configure_pipeline(multiprocessing_manager, args.opt_input_dir)

        serializer = ConcurrentJSONAnalysisSerializer(analyses_output_fpath, multiprocessing_manager)
        worker_initializer_args = (analyses_namespace, serializer)
        # worker_initializer_args = (None, None)#, domains_info, one_struct_analyses_done_set)

        # run_analyses_multiprocess(args.pairs_json, process_pair_with_exc_handling, args.workers, worker_initializer,
        #                           worker_initializer_args)
        run_analyses_serial(args.pairs_json, process_pair_with_exc_handling, worker_initializer,
                                  worker_initializer_args)

        serializer.dump_data()


if __name__ == '__main__':
    main()