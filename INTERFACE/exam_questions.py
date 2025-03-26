class ExamQuestions:
    def __init__(self):
        self.questions = [
            "What are the requirements for patentability according to Article 52 EPC?",
            "Explain the concept of inventive step in patent law.",
            "What is the difference between PCT and EPC applications?",
            "How are patent claims interpreted under the Guidelines?",
            "What constitutes prior art under the EPC?",
        ]

    def get_random_question(self):
        import random

        return random.choice(self.questions)
