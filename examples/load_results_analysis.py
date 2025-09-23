from sparc import MethodTester, DataHandler


def main():
    data_handler = DataHandler()
    data = data_handler.load_concatenated_simulated_data('../data/SimulatedData_2x64_30000_10trials.npz', 30000)
    
    tester = MethodTester.load_saved_results(data, "../data/results/simulated/")
    
    tester.print_results()
    tester.compare()
    tester.plot_results()

if __name__ == "__main__":
    main()
