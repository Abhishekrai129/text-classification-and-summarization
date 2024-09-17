import unittest
from text_processing import classify_text_bert, summarize_text_t5  # Ensure your Python file is named `text_processing.py`

class TestTextProcessing(unittest.TestCase):
    
    def test_classify_text_bert(self):
        text = "I love this product! It is amazing and worth every penny."
        class_label = classify_text_bert(text)
        self.assertIsInstance(class_label, int)  # Ensure the output is an integer
        
    def test_summarize_text_t5(self):
        text = """
        The United Nations is an international organization founded in 1945. It is currently made up of 193 Member States. The mission and work of the United Nations are guided by the purposes and principles contained in its founding Charter. 
        The United Nations was established after the Second World War by the Allied powers, as a successor to the ineffective League of Nations. The organization's objectives are to maintain international peace and security, promote human rights, foster social and economic development, and coordinate international cooperation.
        """
        summary = summarize_text_t5(text)
        self.assertIsInstance(summary, str)  # Ensure the output is a string
        self.assertGreater(len(summary), 0)  # Ensure the summary is not empty

if __name__ == '__main__':
    unittest.main()
