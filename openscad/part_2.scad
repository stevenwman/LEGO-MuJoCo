w = 0.25;
h = 0.6;

intersection() {
    import("robots/duplo_hip_offset_mjcf/part_1.stl");
    linear_extrude(height=10) {
        polygon(points=[[0,-h/2], [w,-h/2], [w,h/2], [0, h/2]]);
    }
}
