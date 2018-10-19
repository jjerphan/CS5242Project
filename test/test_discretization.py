from discretization import RelativeCubeRepresentation


def compare_relative_cube_discretization(example_file):
    print(f"System {example_file}")
    file_name = os.path.join(training_examples_folder, example_file)
    example = load_nparray(file_name)

    print("Relative Compressed Representation")
    cube = RelativeCubeRepresentation(length_cube_side,
                                      use_rotation_invariance=False,
                                      keep_proportions=False,
                                      verbose=True).make_cube(example)
    plot_cube(cube)
    mngr = plt.get_current_fig_manager()
    mngr.window.setGeometry(0, 100, 640, 545)
    plt.pause(0.003)

    print("Relative Properly Scaled Representation")
    cube = make_relative_cube(example, length_cube_side=length_cube_side,
                              use_rotation_invariance=False,
                              keep_proportions=True,
                              verbose=True)
    plot_cube(cube)
    mngr = plt.get_current_fig_manager()
    mngr.window.setGeometry(640, 100, 640, 545)
    plt.pause(1)

    print("Relative Properly Scaled Rotation Invariant Representation")
    cube = make_relative_cube(example, length_cube_side=length_cube_side,
                              use_rotation_invariance=True,
                              keep_proportions=True,
                              verbose=True)

    plot_cube(cube)
    mngr = plt.get_current_fig_manager()
    mngr.window.setGeometry(2 * 640, 100, 640, 545)
    plt.pause(2)


def compare_absolute_cube_discretization(example_file, res_cube=1.0):
    print(f"System {example_file}")
    file_name = os.path.join(training_examples_folder, example_file)
    example = load_nparray(file_name)

    print("Absolute Representation")
    cube = make_absolute_cube(example, length_cube_side=length_cube_side,
                              res_cube=res_cube,
                              use_rotation_invariance=False,
                              verbose=True)
    plot_cube(cube)
    mngr = plt.get_current_fig_manager()
    mngr.window.setGeometry(0, 100, 640, 545)
    plt.pause(0.003)

    print("Absolute Rotation Invariant Representation")
    cube = make_absolute_cube(example, length_cube_side=length_cube_side,
                              res_cube=res_cube,
                              use_rotation_invariance=True,
                              verbose=True)

    plot_cube(cube)
    mngr = plt.get_current_fig_manager()
    mngr.window.setGeometry(640, 100, 640, 545)
    plt.pause(2)


if __name__ == "__main__":
    # Just to test
    examples_files = sorted(os.listdir(training_examples_folder))
    plt.ion()
    plt.show()
    for ex_file in examples_files[:3]:
        plt.close('all')
        compare_relative_cube_discretization(ex_file)
        compare_absolute_cube_discretization(ex_file, res_cube=3.0)
        input("Show next [Press Enter]")
