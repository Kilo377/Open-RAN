"""RV0 circular-buffer repetition for single-CB low-rate NR reference codes.

Sionna 2.2's LDPC5GEncoder rejects effective rates below 1/5. We retain its
BG2 mother encoder, CRC, scrambler and BP decoder, adding circular selection
over the punctured, filler-free mother buffer. No package/global patching.
Multi-CB repetition remains explicitly unsupported. This is equivalent-link
RV0 coding, not a claim of waveform HARQ soft combining.
"""
import numpy as np


class LowRateNRCodec:
    def __init__(self, bits, qm, coded_bits, device):
        import torch
        from sionna.phy.fec.crc import CRCEncoder, CRCDecoder
        from sionna.phy.fec.ldpc import LDPC5GEncoder, LDPCBPDecoder
        from sionna.phy.fec.scrambling import TB5GScrambler, Descrambler
        self.t, self.device, self.bits = torch, device, int(bits)
        if bits > 3824:
            raise ValueError('Repetition coding is currently not supported for multi-CB reference')
        self.crc = CRCEncoder('CRC16', k=bits, device=device)
        self.crc_decoder = CRCDecoder(self.crc, device=device)
        k = bits + 16
        self.mother = LDPC5GEncoder(k, 3*k, bg='bg2', device=device)
        self.bp = LDPCBPDecoder(self.mother.pcm, num_iter=20, hard_out=True, device=device)
        # Map compressed circular buffer positions back to mother-code indices.
        indices = np.r_[np.arange(2*self.mother.z, k), np.arange(self.mother.k_ldpc, self.mother.n_ldpc)]
        permutation, _ = self.mother.generate_out_int(coded_bits, qm)
        self.indices = torch.as_tensor(indices[np.arange(coded_bits) % len(indices)][permutation], dtype=torch.int64, device=device)
        self.scrambler = TB5GScrambler(n_rnti=1, n_id=1, binary=True, channel_type='PDSCH', device=device)
        self.descrambler = Descrambler(self.scrambler, binary=False, device=device)
        self.k = k

    def encode(self, bits):
        t = self.t
        crc_bits = self.crc(bits)
        # This pinned-version hook generates the exact Sionna BG2 mother word;
        # the standard encoder's public output has already discarded parity.
        padded = t.cat([crc_bits, t.zeros(*bits.shape[:-1], self.mother.k_ldpc-self.k, device=self.device)], dim=-1)
        mother = self.mother._encode_fast(padded.reshape(-1, self.mother.k_ldpc)).reshape(*bits.shape[:-1], self.mother.n_ldpc)
        return self.scrambler(mother[..., self.indices])

    def decode(self, llr):
        t = self.t
        llr = self.descrambler(llr)
        full = t.zeros(*llr.shape[:-1], self.mother.n_ldpc, device=self.device)
        full.scatter_add_(-1, self.indices.expand_as(llr), llr)
        # Known-zero filler bits; punctured bits retain zero evidence.
        full[..., self.k:self.mother.k_ldpc] = -100.
        decoded = self.bp(full)[..., :self.k]
        bits, valid = self.crc_decoder(decoded)
        return bits, valid.squeeze(-1)
