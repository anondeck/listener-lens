#!/usr/bin/env python3
from earshift_bakeoff.deploy_service import run


if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        pass
