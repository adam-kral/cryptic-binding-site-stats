from functools import partial, lru_cache
from pathlib import Path

from apo_holo_structure_stats.core.analyses import GetChains, GetCAlphaCoords, GetCentroid, GetCenteredCAlphaCoords, \
    GetRotationMatrix, GetHingeAngle, GetRMSD, GetSecondaryStructureForStructure, CompareSecondaryStructure, \
    GetDomainsForStructure, GetSASAForStructure, GetInterfaceBuriedArea
from apo_holo_structure_stats.pipeline.run_analyses_settings import SavedAnalysis, dotdict, mulproc_lru_cache, \
    parse_mmcif_exp_method_hack, simplify_args_decorator


def configure_pipeline(manager, input_dir: Path):
    # todo all other analyses could be cached at least with lru_cache for the time a pair is processed (they are called
    # multiple times in some cases..)
    # def end_pair_callback():
    #     for cache in caches:
    #         cache.cache_clear()



    # todo delete this
    #  need to skip non-xray now (we still have old data unskipped in `filter_structures`)
    not_xray = manager.dict()
    # parse_mmcif = mulproc_lru_cache(partial(parse_mmcif_exp_method_hack, not_xray), manager, max_size=2)
    parse_mmcif = lru_cache(maxsize=3)(simplify_args_decorator(partial(parse_mmcif_exp_method_hack, {})))
    # parse_mmcif = mulproc_lru_cache(partial(parse_mmcif, allow_download=False), manager, max_size=2)

    get_chains = GetChains()

    get_c_alpha_coords = GetCAlphaCoords()
    get_centroid = GetCentroid((get_c_alpha_coords,))
    get_centered_c_alpha_coords = GetCenteredCAlphaCoords((get_c_alpha_coords, get_centroid))
    get_rotation_matrix = GetRotationMatrix((get_centered_c_alpha_coords,))

    get_hinge_angle = GetHingeAngle((get_c_alpha_coords, get_centroid, get_rotation_matrix))
    get_rmsd = GetRMSD((get_centered_c_alpha_coords, get_rotation_matrix))

    # get_ss = GetSecondaryStructureForStructure()
    # get_ss = mulproc_lru_cache(get_ss, manager, MP_CACHE_SIZE)
    get_ss = SavedAnalysis(str(input_dir / 'db_get_ss'), GetSecondaryStructureForStructure())
    cmp_ss = CompareSecondaryStructure((get_ss,))

    get_domains = SavedAnalysis(str(input_dir / 'db_get_domains'), GetDomainsForStructure())

    get_sasa = GetSASAForStructure()
    # nepouzivat cache, protoze hranice domen se mohou menit podle partnera v páru (unobserved nebo cropped protoze
    # lcs)
    # get_sasa = mulproc_lru_cache(get_sasa, manager, MP_CACHE_SIZE)
    get_interdomain_surface = GetInterfaceBuriedArea((get_sasa,))

    analyzer_namespace = locals()
    del analyzer_namespace['manager']  # remove the argument from the namespace
    del analyzer_namespace['input_dir']  # remove the argument from the namespace

    return analyzer_namespace