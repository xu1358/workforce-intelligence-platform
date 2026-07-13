import sys

import numpy as np
import pandas as pd
from faker import Faker


def main() -> None:
    """Verify that the basic project environment works."""

    fake = Faker()

    test_data = {
        "employee_id": [1, 2, 3],
        "employee_name": [
            fake.name(),
            fake.name(),
            fake.name(),
        ],
        "monthly_salary": np.array([5000, 6200, 7100]),
    }

    employees = pd.DataFrame(test_data)

    print("Python version:")
    print(sys.version)

    print("\nTest employee table:")
    print(employees)

    print("\nAverage monthly salary:")
    print(employees["monthly_salary"].mean())

    print("\nEnvironment setup completed successfully.")


if __name__ == "__main__":
    main()