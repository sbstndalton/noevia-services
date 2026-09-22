import unittest
from server import validate

class ValidationTests(unittest.TestCase):
    def test_valid(self):
        body = {'state':'synthetic','question':'Next?', 'options':[{'id':'continue','label':'Continue'}, {'id':'verify','label':'Check'}]}
        self.assertEqual(validate(body), body)
    def test_invalid(self):
        for body in [None, {}, {'state':'x','question':'?', 'options':[]}, {'state':'x','question':'?', 'options':[{'id':'x','label':'X'}]*2}]:
            with self.assertRaises(ValueError): validate(body)

if __name__ == '__main__': unittest.main()
