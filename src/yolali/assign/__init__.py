"""Label assignment: which anchors are responsible for which ground truth."""

from yolali.assign.o2o import OneToOneAssigner
from yolali.assign.tal import Assignment, TaskAlignedAssigner

__all__ = ["Assignment", "OneToOneAssigner", "TaskAlignedAssigner"]
