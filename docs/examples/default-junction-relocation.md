# What junction relocation changes

Since delaunay-watershed-3d 0.5.0, every reconstruction moves the trijunction vertices onto the junction curves the mask gives. [`Examples/default_junction_relocation.py`](https://github.com/VirtualEmbryo/foambryo/blob/v0.7.0/Examples/default_junction_relocation.py) reconstructs a label image twice, with and without the relocation, infers the tensions of both meshes and draws the junction line the relocation moved most. Run it on your own label image (medium 0); it needs scikit-image to read it:

```shell
python Examples/default_junction_relocation.py path/to/labels.tif --out junction.png
```

```python
--8<-- "Examples/default_junction_relocation.py"
```
