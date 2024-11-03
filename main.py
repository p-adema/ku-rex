import global_state
from path_plan import main_thread

try:
    main_thread()
finally:
    global_state.stop_program.set()
    print("Main thread exiting...")
