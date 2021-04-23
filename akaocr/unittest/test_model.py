# -*- coding: utf-8 -*-
"""
_____________________________________________________________________________
Created By  : Nguyen Viet Bac - Bacnv6
Created Date: Tue December 29 13:00:00 VNT 2020
Project : AkaOCR core
_____________________________________________________________________________

This file contain config methods
_____________________________________________________________________________
"""
import sys
sys.path.append("../")


import unittest
from unittest import mock
from testcases.config_test import config_test
from testcases.models_test import main as main_model
#config_test, models_test, test_dataloader, train_loop_test, evaluation_test

class TestModel(unittest.TestCase):
    
    def test_config(self):
        cf_recog, cf_detec = config_test()
        self.assertIsNotNone(cf_recog, cf_detec)

    def test_model(self):
        mdl_detec_y0, mdl_detec_y1, mdl_recog = main_model()
        self.assertIsNotNone(mdl_detec_y0, mdl_detec_y1)
        self.assertIsNotNone(mdl_recog)

if __name__ == '__main__':
    unittest.main(verbosity=2)