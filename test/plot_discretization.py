import os

from matplotlib import pyplot as plt
from discretization import RelativeCubeRepresentation, load_nparray, AbsoluteCubeRepresentation
from settings import training_examples_folder, length_cube_side


def compare_relative_cube_discretization(example_file):
    print(f"System {example_file}")
    file_name = os.path.join(training_examples_folder, example_file)
    example = load_nparray(file_name)

    print("Relative Compressed Representation")
    rel_compressed_repr = RelativeCubeRepresentation(length_cube_side=length_cube_side,
                                                     use_rotation_invariance=False,
                                                     keep_proportions=False,
                                                     verbose=True)

    cube = rel_compressed_repr.make_cube(example)
    rel_compressed_repr.plot_cube(cube)

    mngr = plt.get_current_fig_manager()
    mngr.window.setGeometry(0, 100, 640, 545)
    plt.pause(0.003)

    print("Relative Properly Scaled Representation")
    rel_prop_scaled_representation = RelativeCubeRepresentation(length_cube_side=length_cube_side,
                                                                use_rotation_invariance=False,
                                                                keep_proportions=True,
                                                                verbose=True)
    cube = rel_prop_scaled_representation.make_cube(example)
    rel_prop_scaled_representation.plot_cube(cube)
    mngr = plt.get_current_fig_manager()
    mngr.window.setGeometry(640, 100, 640, 545)
    plt.pause(1)

    print("Relative Properly Scaled Rotation Invariant Representation")
    rel_prop_scaled_rot_inv_representation = RelativeCubeRepresentation(length_cube_side=length_cube_side,
                                                                        use_rotation_invariance=True,
                                                                        keep_proportions=True,
                                                                        verbose=True)
    cube = rel_prop_scaled_rot_inv_representation.make_cube(example)

    rel_prop_scaled_rot_inv_representation.plot_cube(cube)
    mngr = plt.get_current_fig_manager()
    mngr.window.setGeometry(2 * 640, 100, 640, 545)
    plt.pause(2)


def compare_absolute_cube_discretization(example_file, cube_resolution=1.0):
    print(f"System {example_file}")
    file_name = os.path.join(training_examples_folder, example_file)
    example = load_nparray(file_name)

    print("Absolute Representation")
    abs_repr = AbsoluteCubeRepresentation(length_cube_side=length_cube_side,
                                          cube_resolution=cube_resolution,
                                          use_rotation_invariance=False,
                                          verbose=True)

    cube = abs_repr.make_cube(example)
    abs_repr.plot_cube(cube)
    mngr = plt.get_current_fig_manager()
    mngr.window.setGeometry(0, 100, 640, 545)
    plt.pause(0.003)

    print("Absolute Rotation Invariant Representation")
    abs_repr_rot_inv = AbsoluteCubeRepresentation(length_cube_side=length_cube_side,
                                                  cube_resolution=cube_resolution,
                                                  use_rotation_invariance=True,
                                                  verbose=True)
    cube = abs_repr_rot_inv.make_cube(example)

    abs_repr_rot_inv.plot_cube(cube)
    mngr = plt.get_current_fig_manager()
    mngr.window.setGeometry(640, 100, 640, 545)
    plt.pause(2)


if __name__ == "__main__":
    # Just to test
    examples_files = os.listdir(training_examples_folder)
    plt.ion()
    plt.show()
    for ex_file in examples_files[:10]:
        plt.close('all')
        compare_relative_cube_discretization(ex_file)
        compare_absolute_cube_discretization(ex_file, cube_resolution=3.0)
        input("Show next [Press Enter]")
