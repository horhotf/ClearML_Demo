import os
from clearml import Task

from task_stats_to_comment import get_clearml_task_of_current_commit


def compare_and_tag_task(commit_hash):
    """Compare current performance to best previous performance and only allow equal or better."""
    current_task = get_clearml_task_of_current_commit(commit_hash)
    best_task = Task.get_task(project_name='Titanic_cicd', task_name='Titanic_cicd', tags=['Best Performance'])
    if best_task:
        best_metric = min(
            best_task.get_reported_scalars().get('Logloss').get('Test').get('y')
        )
        current_metric = min(
            current_task.get_reported_scalars().get('Logloss').get('Test').get('y')
        )
        print(f"Best metric in the system is: {best_metric} and current metric is {current_metric}")
        if current_metric <= best_metric:
            print("This means current metric is better or equal! Tagging as such.")
            current_task.add_tags(['Best Performance'])
        else:
            print("This means current metric is worse! Not tagging.")
    else:
        current_task.add_tags(['Best Performance'])


if __name__ == '__main__':
    print(f"Running on commit hash: {os.getenv('COMMIT_ID')}")
    compare_and_tag_task(os.getenv('COMMIT_ID'))