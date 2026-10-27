"""Exercise all original generators through the new transport on private fixtures."""
import importlib.util
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
OLD=HERE.parents[1]
NEW=OLD.parent/'recogdrive_dc8_native_candidates_20261027'
sys.path.insert(0,str(NEW/'code'))
import gt_opt_controller as new_controller
from gt_numeric import atomic

spec=importlib.util.spec_from_file_location('original_gt_generator_fixture',OLD/'engineering/preflight_gt_stage_generators.py')
fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)
fixture.controller=new_controller
original_fixture=fixture.fixture_root
def fixture_root(root,source):
    scenarios=original_fixture(root,source)
    atomic(root/'PROTOCOL.json',{'reuse_old_intersection_indices':[],'local_candidates':{'synthetic_only':True}})
    return scenarios
fixture.fixture_root=fixture_root

if __name__=='__main__':
    sys.argv=[__file__,'--root',str(HERE/'ALL_STAGE_GENERATORS_PRIVATE_FIXTURES'),
              '--source-root',str(OLD/'source_snapshot/ReCogDrive')]
    fixture.main()
