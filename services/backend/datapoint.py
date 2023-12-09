from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List
from uuid import uuid4
import random


@dataclass
class Datapoint:

    id: int
    high_d_vector: List[float]
    low_d_vector: List[float]
    label: str | None
    is_landmark: bool
    data: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "high_d_vector": self.high_d_vector,
            "low_d_vector": self.low_d_vector,
            "label": self.label,
            "is_landmark": self.is_landmark,
            "data": self.data,
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> Datapoint:
        return Datapoint(
            id=data["id"],
            high_d_vector=data["high_d_vector"],
            low_d_vector=data["low_d_vector"],
            label=data["label"],
            is_landmark=data["is_landmark"],
            data=data["data"],
        )

    @classmethod
    def from_dict_bulk(cls, data: List[Dict[str, Any]]) -> List[Datapoint]:
        return [cls.from_dict(d) for d in data]

    @classmethod
    def generate_random(
        cls,
        id: int,
        high_d_vector_size: int,
        low_d_vector_size: int,
        labels: List[str] | None = None,
        landmark_ratio: float = 1.0,
        generate_random_data: bool = False
    ) -> Datapoint:
        labels = labels or [None]
        return cls(
            id=id,
            high_d_vector=[random.random() for _ in range(high_d_vector_size)],
            low_d_vector=[random.random() for _ in range(low_d_vector_size)],
            label=random.choice(labels),
            is_landmark=random.random() < landmark_ratio,
            data={} if not generate_random_data else {
                "random_data": str(uuid4())
            }
        )

    @classmethod
    def generate_random_bulk(
        cls,
        amount: int,
        high_d_vector_size: int,
        low_d_vector_size: int,
        labels: List[str] | None = None,
        landmark_ratio: float = 1.0,
        generate_random_data: bool = False
    ) -> List[Datapoint]:
        return [
            cls.generate_random(
                id=i,
                high_d_vector_size=high_d_vector_size,
                low_d_vector_size=low_d_vector_size,
                labels=labels,
                landmark_ratio=landmark_ratio,
                generate_random_data=generate_random_data
            )
            for i in range(amount)
        ]
