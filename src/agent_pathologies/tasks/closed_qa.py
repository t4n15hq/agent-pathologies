from __future__ import annotations

import random

from ..types import Role, Turn
from .base import Task, TaskInstance
from .scoring import score_letter


# Curated unambiguous multiple-choice trivia. Letter is the correct answer.
# Designed so a competent model scores ~95%+ at baseline, leaving headroom
# for perturbation effects to be visible.
_QUESTIONS: list[tuple[str, list[str], str]] = [
    ("In what year did humans first land on the Moon?",
     ["1965", "1967", "1969", "1971"], "C"),
    ("Which planet has the most moons (as of standard astronomy references)?",
     ["Jupiter", "Saturn", "Uranus", "Neptune"], "B"),
    ("Which gas makes up the largest fraction of Earth's atmosphere?",
     ["Oxygen", "Carbon dioxide", "Nitrogen", "Argon"], "C"),
    ("Who wrote the novel 'One Hundred Years of Solitude'?",
     ["Mario Vargas Llosa", "Gabriel Garcia Marquez", "Isabel Allende", "Jorge Luis Borges"], "B"),
    ("What is the speed of light in a vacuum (approximate)?",
     ["3.0 × 10^6 m/s", "3.0 × 10^7 m/s", "3.0 × 10^8 m/s", "3.0 × 10^9 m/s"], "C"),
    ("In what year did the Berlin Wall fall?",
     ["1987", "1988", "1989", "1991"], "C"),
    ("Which element has the chemical symbol 'Au'?",
     ["Silver", "Aluminum", "Gold", "Argon"], "C"),
    ("What is the largest ocean on Earth?",
     ["Atlantic", "Indian", "Arctic", "Pacific"], "D"),
    ("Who painted the ceiling of the Sistine Chapel?",
     ["Leonardo da Vinci", "Raphael", "Michelangelo", "Donatello"], "C"),
    ("Which language has the most native speakers worldwide?",
     ["English", "Spanish", "Hindi", "Mandarin Chinese"], "D"),
    ("What is the boiling point of water at sea level in Celsius?",
     ["90", "95", "100", "105"], "C"),
    ("Which country is both in Europe and Asia by geography?",
     ["Greece", "Turkey", "Egypt", "Israel"], "B"),
    ("Who developed the theory of general relativity?",
     ["Isaac Newton", "Niels Bohr", "Albert Einstein", "Max Planck"], "C"),
    ("What is the smallest prime number?",
     ["0", "1", "2", "3"], "C"),
    ("Which year did the Soviet Union dissolve?",
     ["1989", "1990", "1991", "1992"], "C"),
    ("Which is the longest river in the world (by most measures)?",
     ["Amazon", "Nile", "Yangtze", "Mississippi"], "B"),
    ("What is the powerhouse of the cell?",
     ["Nucleus", "Ribosome", "Mitochondrion", "Golgi apparatus"], "C"),
    ("Who is credited with discovering penicillin?",
     ["Louis Pasteur", "Alexander Fleming", "Robert Koch", "Joseph Lister"], "B"),
    ("Which mountain is the tallest above sea level?",
     ["K2", "Kangchenjunga", "Everest", "Lhotse"], "C"),
    ("What does DNA stand for?",
     ["Diribonucleic acid", "Deoxyribonucleic acid", "Dynamic nucleotide assembly", "Dual-nuclear acid"], "B"),
    ("Which empire was ruled by Genghis Khan?",
     ["Ottoman", "Persian", "Mongol", "Byzantine"], "C"),
    ("Which is a noble gas?",
     ["Nitrogen", "Hydrogen", "Helium", "Oxygen"], "C"),
    ("Who wrote 'Hamlet'?",
     ["Charles Dickens", "William Shakespeare", "Geoffrey Chaucer", "John Milton"], "B"),
    ("Approximately how many bones are in the adult human body?",
     ["106", "186", "206", "306"], "C"),
    ("Which planet is known as the Red Planet?",
     ["Venus", "Mars", "Jupiter", "Mercury"], "B"),
]


class ClosedQA(Task):
    """Multiple-choice factual QA. Scored on the letter choice only —
    completely unambiguous."""

    name = "closed_qa_v1"

    def __init__(self) -> None:
        self._questions = _QUESTIONS

    def sample(self, seed: int) -> TaskInstance:
        rng = random.Random(seed)
        question, choices, answer = rng.choice(self._questions)
        letters = "ABCD"
        rendered = "\n".join(f"  {letters[i]}) {c}" for i, c in enumerate(choices))

        system = Turn(
            role=Role.SYSTEM,
            content=(
                "You are answering a multiple-choice question. Respond with "
                "exactly one sentence in the form 'The answer is X.' where "
                "X is the single letter A, B, C, or D."
            ),
        )
        probe = Turn(role=Role.USER, content=f"{question}\n{rendered}")
        answer_slot = Turn(role=Role.ASSISTANT, content="", is_probe=True)

        return TaskInstance(
            task_id=f"qa-{seed}",
            task_name=self.name,
            setup_turns=[system, probe, answer_slot],
            probe_question=question,
            correct_answer=answer,
            scorer=score_letter,
            metadata={"question": question, "choices": choices, "letter": answer},
        )
