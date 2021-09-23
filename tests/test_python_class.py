def test_static_method():
    class Base:
        @staticmethod
        def say() -> int:
            return 1

    class A(Base):
        pass

    class B(Base):
        pass

    a = A()
    b = B()

    assert a.say() == b.say()


def test_class_method():
    class Base:
        field = None

        @classmethod
        def say(cls) -> str:
            return cls.__name__ + cls.field

    class A(Base):
        field = 'a'

    class B(Base):
        field = 'b'

    a = A()
    b = B()

    assert a.say() == 'Aa'
    assert b.say() == 'Bb'
