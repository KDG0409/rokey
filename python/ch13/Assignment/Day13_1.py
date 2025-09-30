try:
    x = int("abc")
except ValueError:
    print("ValueError occurred!")
finally:
    print("Execution finished.")
