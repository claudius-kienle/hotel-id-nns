import unittest
import torchvision
import torch

from hotel_id_nns.nn.modules.resnet_chris import ResNet, resnet50_cfg

class TestResNets(unittest.TestCase):

    def test_resnet50(self):
        num_classes = 100
        tv_model = torchvision.models.resnet50()
        tv_model.fc = torch.nn.Linear(tv_model.fc.in_features, num_classes)
        our_model = ResNet(network_cfg=resnet50_cfg, out_features=num_classes)

        random_input = torch.rand(size=(1, 3, 244, 244))

        tv_out = tv_model(random_input)
        our_out = our_model(random_input)

        torch.allclose(tv_out, our_out)