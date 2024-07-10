import unittest
from lm_eval.models.vicuna import Vicuna7b
from lm_eval.api.instance import Instance

class TestVicuna7b(unittest.TestCase):
    def setUp(self):
        self.model = Vicuna7b()

    def test_loglikelihood(self):
        requests = [
            Instance("loglikelihood", {}, ("Hello, how are you?", " I'm fine, thank you."), 0)
        ]
        result = self.model.loglikelihood(requests)
        self.assertIsInstance(result, list)

    def test_loglikelihood_rolling(self):
        requests = [
            Instance("loglikelihood_rolling", {}, ("Hello, how are you?",), 0)
        ]
        result = self.model.loglikelihood_rolling(requests)
        self.assertIsInstance(result, list)

    def test_generate_until(self):
        requests = [
            Instance("generate_until", {}, ("Once upon a time", {"max_length": 50}), 0)
        ]
        result = self.model.generate_until(requests)
        self.assertIsInstance(result, list)

    def test_apply_chat_template(self):
        chat_history = [
            {"user": "Hello, how are you?"},
            {"assistant": "I'm fine, thank you."},
            {"user": "What's the weather like today?"}
        ]
        chat_text = self.model.apply_chat_template(chat_history)
        self.assertIsInstance(chat_text, str)

if __name__ == "__main__":
    unittest.main()
