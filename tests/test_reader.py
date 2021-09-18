import os
import sys
from pytest import fixture

from src.schema import Config
from src.dataset_reader.snips import SnipsDataSetReader

root_dir = os.path.dirname(os.path.dirname(__file__))


@fixture(scope='module')
def config():
    sys.argv = [
        '',
        '--dataset', 'snips',
        '--metric', 'l2',
        '--n_way_train', '5',
        '--n_way_validation', '12',
        
        '--k_shot_train', '5',
        '--k_shot_validation', '12'
    ]
    return Config().parse_args(known_only=True)


def test_snips_dataset_reader(config: Config):
    train_file = os.path.join(root_dir, 'data/snips/intent-classification/snips_test.json.corpus')
    reader = SnipsDataSetReader(config)
    
    examples = reader.read(train_file)
    assert len(examples) > 0
