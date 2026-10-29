"""
cpm.py - Critical Path Method (CPM) for Public-Health Interventions
===================================================================
Author: Member 6 (Temporal Planning, CPM & Replanning)
Course: CS4L002 Artificial Intelligence, IIT Bhubaneswar

Implements classical Activity-On-Node Critical Path Method:
1. Forward Pass: Computes Earliest Start (ES) and Earliest Finish (EF).
     ES(j) = max_{i in Pred(j)} EF(i)
     EF(j) = ES(j) + Duration(j)
2. Backward Pass: Computes Latest Finish (LF) and Latest Start (LS).
     LF(i) = min_{j in Succ(i)} LS(j)
     LS(i) = LF(i) - Duration(i)
3. Total Slack (Float): Slack(i) = LS(i) - ES(i) = LF(i) - EF(i).
     Tasks with Slack == 0 form the Critical Path governing total project duration.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any, Set


@dataclass
class CPMTask:
    """Represents an intervention activity node in the CPM network."""
    task_id: str
    duration: float
    dependencies: List[str] = field(default_factory=list)
    zone: str = ""
    action_type: str = ""
    # Computed schedule parameters
    es: float = 0.0
    ef: float = 0.0
    ls: float = 0.0
    lf: float = 0.0
    slack: float = 0.0
    is_critical: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "zone": self.zone,
            "action_type": self.action_type,
            "duration": self.duration,
            "dependencies": self.dependencies,
            "ES": round(self.es, 1),
            "EF": round(self.ef, 1),
            "LS": round(self.ls, 1),
            "LF": round(self.lf, 1),
            "slack": round(self.slack, 1),
            "is_critical": self.is_critical
        }


class CriticalPathAnalyzer:
    """Calculates CPM early/late schedules and extracts the critical path."""

    @staticmethod
    def compute_critical_path(tasks: List[CPMTask]) -> Tuple[List[CPMTask], List[str], float]:
        """
        Executes Forward and Backward passes over the task dependency network.
        Returns: (scheduled_tasks, critical_path_task_ids, total_project_duration)
        """
        if not tasks:
            return [], [], 0.0

        task_dict: Dict[str, CPMTask] = {t.task_id: t for t in tasks}
        successors: Dict[str, List[str]] = {t.task_id: [] for t in tasks}

        for t in tasks:
            for dep in t.dependencies:
                if dep in successors:
                    successors[dep].append(t.task_id)

        # 1. Forward Pass (Topological order)
        # Compute in-degree
        in_degree = {t.task_id: len([d for d in t.dependencies if d in task_dict]) for t in tasks}
        queue = [tid for tid, deg in in_degree.items() if deg == 0]
        topo_order = []

        while queue:
            curr_id = queue.pop(0)
            topo_order.append(curr_id)
            curr_task = task_dict[curr_id]

            # ES is max of predecessor EFs
            if not curr_task.dependencies:
                curr_task.es = 0.0
            else:
                pred_efs = [task_dict[d].ef for d in curr_task.dependencies if d in task_dict]
                curr_task.es = max(pred_efs) if pred_efs else 0.0

            curr_task.ef = curr_task.es + curr_task.duration

            for succ_id in successors[curr_id]:
                in_degree[succ_id] -= 1
                if in_degree[succ_id] == 0:
                    queue.append(succ_id)

        project_duration = max((t.ef for t in tasks), default=0.0)

        # 2. Backward Pass (Reverse topological order)
        for tid in reversed(topo_order):
            curr_task = task_dict[tid]
            succ_ids = successors[tid]

            if not succ_ids:
                curr_task.lf = project_duration
            else:
                succ_lss = [task_dict[s].ls for s in succ_ids]
                curr_task.lf = min(succ_lss) if succ_lss else project_duration

            curr_task.ls = curr_task.lf - curr_task.duration
            curr_task.slack = max(0.0, curr_task.ls - curr_task.es)
            curr_task.is_critical = abs(curr_task.slack) < 1e-4

        critical_path = [t.task_id for t in tasks if t.is_critical]
        return tasks, critical_path, project_duration
