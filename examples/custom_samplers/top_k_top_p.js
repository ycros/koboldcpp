function sample(s) {
    s.topK(80);
    s.topP(0.92, 1);
    return s.pick();
}
