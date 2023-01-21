import unittest
import torchvision
import torch

from hotel_id_nns.nn.modules.resnet_chris import ResNet, resnet50_cfg, resnet18_cfg
from hotel_id_nns.utils.load_torchvision_weights import map_weights

class TestResNets(unittest.TestCase):

    def test_resnet50(self):
        num_classes = 100
        tv_model = torchvision.models.resnet50()
        tv_model.fc = torch.nn.Linear(tv_model.fc.in_features, num_classes)
        our_model = ResNet(network_cfg=resnet50_cfg, out_features=num_classes)
        our_model.load_state_dict(map_weights(tv_model.state_dict()))

        # with open("tv.txt", 'w') as f:
        #     f.write(str(tv_model))

        # with open("our.txt", 'w') as f:
        #     f.write(str(our_model))

        random_input = torch.rand(size=(1, 3, 244, 244))

        tv_out = tv_model(random_input)
        our_out = our_model(random_input)

        self.assertTrue(torch.allclose(tv_out, our_out))

    def test_resnet18(self):
        num_classes = 100
        tv_model = torchvision.models.resnet18()
        tv_model.fc = torch.nn.Linear(tv_model.fc.in_features, num_classes)
        our_model = ResNet(network_cfg=resnet18_cfg, out_features=num_classes)
        mapped_weights = map_weights(tv_model.state_dict())
        our_model.load_state_dict(mapped_weights)

        # with open("tv.txt", 'w') as f:
        #     f.write(str(tv_model))

        # with open("our.txt", 'w') as f:
        #     f.write(str(our_model))

        random_input = torch.rand(size=(1, 3, 244, 244))

        tv_out = tv_model(random_input)
        our_out = our_model(random_input)

        self.assertTrue(torch.allclose(tv_out, our_out))