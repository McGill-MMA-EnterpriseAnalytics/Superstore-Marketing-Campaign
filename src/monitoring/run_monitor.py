from src.monitoring.drift_detector import DriftDetector

def main():
    dd = DriftDetector("config.yaml")
    dd.run_all_checks()

if __name__ == "__main__":
    main()
