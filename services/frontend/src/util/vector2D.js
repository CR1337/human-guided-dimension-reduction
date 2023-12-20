export class Vector2D {
    constructor(x, y) {
        this.x = x;
        this.y = y;
    }

    translate(other) {
        return new Vector2D(this.x + other.x, this.y + other.y);
    }

    scale(factor) {
        return new Vector2D(this.x * factor, this.y * factor);
    }

    to3D() {
        return new Vector3D(this.x, this.y, 1.0);
    }
}