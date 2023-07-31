"""

FACETORCH

"""

from facetorch import FaceAnalyzer
from omegaconf import OmegaConf
from torch.nn.functional import cosine_similarity

def load_config(path_to_config="./facetorch/facetorch_config_verifyOnly.yml"):
  return OmegaConf.load(path_to_config)

def init_analyzer(cfg, path_image='./facetorch/demo.jpg'):
  # initialize
  analyzer = FaceAnalyzer(cfg.analyzer)

  # warmup
  response = analyzer.run(
          path_image=path_image,
          batch_size=cfg.batch_size,
          fix_img_size=cfg.fix_img_size,
          return_img_data=False,
          include_tensors=True,
          path_output='./facetorch/im_out.jpg')
  
  return analyzer


def get_verify_vectors_one_face(analyzer, cfg, face_data, tmp_save='./facetorch/tmp_save.jpg'):
  # Save Image to file

  # Get Facetorch output
  response = analyzer.run(
        path_image=tmp_save,
        batch_size=cfg.batch_size,
        fix_img_size=cfg.fix_img_size,
        return_img_data=cfg.return_img_data,
        include_tensors=cfg.include_tensors,
        path_output='./facetorch/im_out.jpg')

return



