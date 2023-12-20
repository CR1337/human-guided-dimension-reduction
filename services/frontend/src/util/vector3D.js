export class Vector3D {
    constructor(x, y, z) {
        this.x = x;
        this.y = y;
        this.z = z;
    }

    translate(other) {
        return new Vector3D(this.x + other.x, this.y + other.y, this.z + other.z);
    }

    scale(factor) {
        return new Vector3D(this.x * factor, this.y * factor, this.z * factor);
    }

    to2D() {
        return new Vector2D(this.x / this.z, this.y / this.z);
    }

    transform(matrix) {
        return new Vector3D(
            this.x * matrix._00 + this.y * matrix._10 + this.z * matrix._20,
            this.x * matrix._01 + this.y * matrix._11 + this.z * matrix._21,
            this.x * matrix._02 + this.y * matrix._12 + this.z * matrix._22
        );
    }
}