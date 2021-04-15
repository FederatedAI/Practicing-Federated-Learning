import math
import sys


class EncodedNumber(object):
    """Represents a float or int encoded for Paillier encryption.

    For end users, this class is mainly useful for specifying precision
    when adding/multiplying an :class:`EncryptedNumber` by a scalar.

    If you want to manually encode a number for Paillier encryption,
    then use :meth:`encode`, if de-serializing then use
    :meth:`__init__`.


    .. note::
        If working with other Paillier libraries you will have to agree on
        a specific :attr:`BASE` and :attr:`LOG2_BASE` - inheriting from this
        class and overriding those two attributes will enable this.

    Notes:
      Paillier encryption is only defined for non-negative integers less
      than :attr:`PaillierPublicKey.n`. Since we frequently want to use
      signed integers and/or floating point numbers (luxury!), values
      should be encoded as a valid integer before encryption.

      The operations of addition and multiplication [1]_ must be
      preserved under this encoding. Namely:

      1. Decode(Encode(a) + Encode(b)) = a + b
      2. Decode(Encode(a) * Encode(b)) = a * b

      for any real numbers a and b.

      Representing signed integers is relatively easy: we exploit the
      modular arithmetic properties of the Paillier scheme. We choose to
      represent only integers between
      +/-:attr:`~PaillierPublicKey.max_int`, where `max_int` is
      approximately :attr:`~PaillierPublicKey.n`/3 (larger integers may
      be treated as floats). The range of values between `max_int` and
      `n` - `max_int` is reserved for detecting overflows. This encoding
      scheme supports properties #1 and #2 above.

      Representing floating point numbers as integers is a harder task.
      Here we use a variant of fixed-precision arithmetic. In fixed
      precision, you encode by multiplying every float by a large number
      (e.g. 1e6) and rounding the resulting product. You decode by
      dividing by that number. However, this encoding scheme does not
      satisfy property #2 above: upon every multiplication, you must
      divide by the large number. In a Paillier scheme, this is not
      possible to do without decrypting. For some tasks, this is
      acceptable or can be worked around, but for other tasks this can't
      be worked around.

      In our scheme, the "large number" is allowed to vary, and we keep
      track of it. It is:

        :attr:`BASE` ** :attr:`exponent`

      One number has many possible encodings; this property can be used
      to mitigate the leak of information due to the fact that
      :attr:`exponent` is never encrypted.

      For more details, see :meth:`encode`.

    .. rubric:: Footnotes

    ..  [1] Technically, since Paillier encryption only supports
      multiplication by a scalar, it may be possible to define a
      secondary encoding scheme `Encode'` such that property #2 is
      relaxed to:

        Decode(Encode(a) * Encode'(b)) = a * b

      We don't do this.


    Args:
      public_key (PaillierPublicKey): public key for which to encode
        (this is necessary because :attr:`~PaillierPublicKey.max_int`
        varies)
      encoding (int): The encoded number to store. Must be positive and
        less than :attr:`~PaillierPublicKey.max_int`.
      exponent (int): Together with :attr:`BASE`, determines the level
        of fixed-precision used in encoding the number.

    Attributes:
      public_key (PaillierPublicKey): public key for which to encode
        (this is necessary because :attr:`~PaillierPublicKey.max_int`
        varies)
      encoding (int): The encoded number to store. Must be positive and
        less than :attr:`~PaillierPublicKey.max_int`.
      exponent (int): Together with :attr:`BASE`, determines the level
        of fixed-precision used in encoding the number.
    """
    BASE = 16
    """Base to use when exponentiating. Larger `BASE` means
    that :attr:`exponent` leaks less information. If you vary this,
    you'll have to manually inform anyone decoding your numbers.
    """
    LOG2_BASE = math.log(BASE, 2)
    FLOAT_MANTISSA_BITS = sys.float_info.mant_dig

    def __init__(self, public_key, encoding, exponent):
        self.public_key = public_key
        self.encoding = encoding
        self.exponent = exponent

    @classmethod
    def encode(cls, public_key, scalar, precision=None, max_exponent=None):
        """Return an encoding of an int or float.

        This encoding is carefully chosen so that it supports the same
        operations as the Paillier cryptosystem.

        If *scalar* is a float, first approximate it as an int, `int_rep`:

            scalar = int_rep * (:attr:`BASE` ** :attr:`exponent`),

        for some (typically negative) integer exponent, which can be
        tuned using *precision* and *max_exponent*. Specifically,
        :attr:`exponent` is chosen to be equal to or less than
        *max_exponent*, and such that the number *precision* is not
        rounded to zero.

        Having found an integer representation for the float (or having
        been given an int `scalar`), we then represent this integer as
        a non-negative integer < :attr:`~PaillierPublicKey.n`.

        Paillier homomorphic arithemetic works modulo
        :attr:`~PaillierPublicKey.n`. We take the convention that a
        number x < n/3 is positive, and that a number x > 2n/3 is
        negative. The range n/3 < x < 2n/3 allows for overflow
        detection.

        Args:
          public_key (PaillierPublicKey): public key for which to encode
            (this is necessary because :attr:`~PaillierPublicKey.n`
            varies).
          scalar: an int or float to be encrypted.
            If int, it must satisfy abs(*value*) <
            :attr:`~PaillierPublicKey.n`/3.
            If float, it must satisfy abs(*value* / *precision*) <<
            :attr:`~PaillierPublicKey.n`/3
            (i.e. if a float is near the limit then detectable
            overflow may still occur)
          precision (float): Choose exponent (i.e. fix the precision) so
            that this number is distinguishable from zero. If `scalar`
            is a float, then this is set so that minimal precision is
            lost. Lower precision leads to smaller encodings, which
            might yield faster computation.
          max_exponent (int): Ensure that the exponent of the returned
            `EncryptedNumber` is at most this.

        Returns:
          EncodedNumber: Encoded form of *scalar*, ready for encryption
          against *public_key*.
        """
        # Calculate the maximum exponent for desired precision
        if precision is None:
            if isinstance(scalar, int):
                prec_exponent = 0
            elif isinstance(scalar, float):
                # Encode with *at least* as much precision as the python float
                # What's the base-2 exponent on the float?
                bin_flt_exponent = math.frexp(scalar)[1]

                # What's the base-2 exponent of the least significant bit?
                # The least significant bit has value 2 ** bin_lsb_exponent
                bin_lsb_exponent = bin_flt_exponent - cls.FLOAT_MANTISSA_BITS

                # What's the corresponding base BASE exponent? Round that down.
                prec_exponent = math.floor(bin_lsb_exponent / cls.LOG2_BASE)
            else:
                raise TypeError("Don't know the precision of type %s."
                                % type(scalar))
        else:
            prec_exponent = math.floor(math.log(precision, cls.BASE))

        # Remember exponents are negative for numbers < 1.
        # If we're going to store numbers with a more negative
        # exponent than demanded by the precision, then we may
        # as well bump up the actual precision.
        if max_exponent is None:
            exponent = prec_exponent
        else:
            exponent = min(max_exponent, prec_exponent)

        int_rep = int(round(scalar * pow(cls.BASE, -exponent)))

        if abs(int_rep) > public_key.max_int:
            raise ValueError('Integer needs to be within +/- %d but got %d'
                             % (public_key.max_int, int_rep))

        # Wrap negative numbers by adding n
        return cls(public_key, int_rep % public_key.n, exponent)

    def decode(self):
        """Decode plaintext and return the result.

        Returns:
          an int or float: the decoded number. N.B. if the number
            returned is an integer, it will not be of type float.

        Raises:
          OverflowError: if overflow is detected in the decrypted number.
        """
        if self.encoding >= self.public_key.n:
            # Should be mod n
            raise ValueError('Attempted to decode corrupted number')
        elif self.encoding <= self.public_key.max_int:
            # Positive
            mantissa = self.encoding
        elif self.encoding >= self.public_key.n - self.public_key.max_int:
            # Negative
            mantissa = self.encoding - self.public_key.n
        else:
            raise OverflowError('Overflow detected in decrypted number')

        return mantissa * pow(self.BASE, self.exponent)

    def decrease_exponent_to(self, new_exp):
        """Return an `EncodedNumber` with same value but lower exponent.

        If we multiply the encoded value by :attr:`BASE` and decrement
        :attr:`exponent`, then the decoded value does not change. Thus
        we can almost arbitrarily ratchet down the exponent of an
        :class:`EncodedNumber` - we only run into trouble when the encoded
        integer overflows. There may not be a warning if this happens.

        This is necessary when adding :class:`EncodedNumber` instances,
        and can also be useful to hide information about the precision
        of numbers - e.g. a protocol can fix the exponent of all
        transmitted :class:`EncodedNumber` to some lower bound(s).

        Args:
          new_exp (int): the desired exponent.

        Returns:
          EncodedNumber: Instance with the same value and desired
            exponent.

        Raises:
          ValueError: You tried to increase the exponent, which can't be
            done without decryption.
        """
        if new_exp > self.exponent:
            raise ValueError('New exponent %i should be more negative than'
                             'old exponent %i' % (new_exp, self.exponent))
        factor = pow(self.BASE, self.exponent - new_exp)
        new_enc = self.encoding * factor % self.public_key.n
        return self.__class__(self.public_key, new_enc, new_exp)