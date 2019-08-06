import datetime
import os

from common import mparams, mx_logging, settings
from common.deployment import deploy_to_gce
from common.util import deploy_util, helper_util
from projects.edison.eval import image_query
from projects.edison.util import predictions_loader
from projects.edison.train import model_preprocessor_specs

from train import train


PARAMS = mparams.make_category(__name__)
# PARAMS.define("output_dir", type=str, help="Point here to an output dir from common.run_train")

def run_eval():
    train()

class CycleDeployer(deploy_to_gce.Deployer):
    def __init__(self, use_gpu_flag):
        super(CycleDeployer, self).__init__()
        self.use_gpu_flag = use_gpu_flag

    def execute(self):
        train()

    def get_gce_instance_config(self):
        instance_name = "se-{}-{}".format(
            os.path.basename("gs://mx-healthcare-derived/experiments/frcnn/190703_derek_hologic_only/".rstrip("/")), datetime.datetime.now().strftime("%y%m%d-%H%M")
        )
        gpu_type = deploy_to_gce.GPUType.P100 if self.use_gpu_flag else None

        return deploy_to_gce.GCEInstanceConfig(
            name=instance_name,
            gpu_type=gpu_type,
            machine_type="n1-highmem-8",
            service_account_email=settings.HEALTHCARE_SERVICE_ACCOUNT,
        )

    def script_file_name(self):
        return deploy_util.prepare_module_name(__file__)


def main():
    CycleDeployer(use_gpu_flag=True).deploy()


if __name__ == "__main__":
    mparams.load_params()
    main()
