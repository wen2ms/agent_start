from dataclasses import dataclass


@dataclass
class UserContext:
    user_id: str
    report: bool = False
