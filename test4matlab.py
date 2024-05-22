import matlab.engine

#def run_matlab_function(input_arg):
#    eng = matlab.engine.start_matlab()  # 启动MATLAB引擎
#    result = eng.test(input_arg, nargout=1)  # 调用函数
#    print('Result from MATLAB:', result)
#    eng.quit()  # 关闭MATLAB引擎




def run_matlab_simulation(a0):
    # Start the MATLAB engine
    eng = matlab.engine.start_matlab()

    # Call the MATLAB function
    # Ensure the function name matches the one defined in MATLAB
    eng.PDE_constraints(float(a0), nargout=0)  # Corrected function name here

    # Close the MATLAB engine
    eng.quit()

if __name__ == "__main__":
    run_matlab_simulation(0.1)