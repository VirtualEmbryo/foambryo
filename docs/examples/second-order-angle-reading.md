# The second-order angle reading

[`Examples/second_order_angle_reading.py`](https://github.com/VirtualEmbryo/foambryo/blob/v0.7.0/Examples/second_order_angle_reading.py) infers the tensions of one benchmark mesh with `angle_reading="chord"` and with `angle_reading="quadric"`, and prints the two contrasts (the slope of inferred against true tensions, 1 is perfect) and the size of the correction the quadric reading applies. It reads the benchmark's ground-truth meshes, `benchmarking-dataset/`, beside the repository; they are not shipped with it.

On those meshes, far finer than any reconstruction, the correction makes the contrast slightly worse: that is the lesson. The reading is meant for coarse reconstructed meshes; see [How the contact angle is read](../user-guide/methods.md#how-the-contact-angle-is-read-angle_reading).

```shell
python Examples/second_order_angle_reading.py --case 000
```

```python
--8<-- "Examples/second_order_angle_reading.py"
```
