
import { EmbeddingVector, CNNHyperParameters, TrainingMetrics } from '../types';

// --- Math Helpers for Vector Operations ---

const add = (v1: number[], v2: number[]) => v1.map((val, i) => val + v2[i]);
const sub = (v1: number[], v2: number[]) => v1.map((val, i) => val - v2[i]);
const mult = (v: number[], scalar: number) => v.map(val => val * scalar);
const dot = (v1: number[], v2: number[]) => v1.reduce((acc, val, i) => acc + val * v2[i], 0);
const norm = (v: number[]) => Math.sqrt(dot(v, v));
const normalize = (v: number[]) => {
    const n = norm(v);
    return n === 0 ? v : mult(v, 1 / n);
};

// Euclidean Distance Squared (mais eficiente para cálculo de gradiente)
const distSq = (v1: number[], v2: number[]) => {
    let sum = 0;
    for (let i = 0; i < v1.length; i++) {
        const d = v1[i] - v2[i];
        sum += d * d;
    }
    return sum;
};

/**
 * Simula o treinamento de uma CNN 1D aplicando Triplet Loss nos embeddings existentes.
 * 
 * L = max(d(A, P) - d(A, N) + margin, 0)
 * 
 * Onde:
 * A (Anchor): Uma entidade
 * P (Positive): Entidade relacionada (mesmo tipo ou alta similaridade de keywords)
 * N (Negative): Entidade não relacionada (tipo diferente)
 */
export const trainCNNWithTripletLoss = async (
    embeddings: EmbeddingVector[],
    params: CNNHyperParameters,
    onEpochComplete: (metrics: TrainingMetrics, updatedEmbeddings: EmbeddingVector[]) => void
) => {
    // 1. Deep copy vectors to avoid mutating state directly during calculation
    let currentEmbeddings = embeddings.map(e => ({ ...e, vector: [...e.vector] }));
    const dim = currentEmbeddings[0].vector.length;

    // AdamW Parameters
    const beta1 = 0.9;
    const beta2 = 0.999;
    const epsilon = 1e-8;
    const weightDecay = 0.01;

    // Adam State (Moment1 and Moment2 for each dimension of each vector)
    // Map ID -> { m: number[], v: number[] }
    const optimizerState: Record<string, { m: number[], v: number[] }> = {};
    currentEmbeddings.forEach(e => {
        optimizerState[e.id] = {
            m: new Array(dim).fill(0),
            v: new Array(dim).fill(0)
        };
    });

    let learningRate = params.learningRate;

    // --- Training Loop ---
    for (let epoch = 1; epoch <= params.epochs; epoch++) {
        let totalLoss = 0;
        let activeTriplets = 0;

        // Shuffle dataset indices
        const indices = Array.from({ length: currentEmbeddings.length }, (_, i) => i)
            .sort(() => Math.random() - 0.5);

        // Batch processing (simulated one-by-one for clarity in JS)
        for (const i of indices) {
            const anchor = currentEmbeddings[i];
            
            // --- Mining Strategy ---
            let positiveIdx = -1;
            let negativeIdx = -1;

            // Find Positive: Same Entity Type OR Keyword Overlap
            const potentialPositives = indices.filter(idx => idx !== i && (
                currentEmbeddings[idx].entityType === anchor.entityType ||
                hasKeywordOverlap(anchor, currentEmbeddings[idx])
            ));

            if (potentialPositives.length > 0) {
                // Random positive for now (could be hardest positive)
                positiveIdx = potentialPositives[Math.floor(Math.random() * potentialPositives.length)];
            }

            // Find Negative based on Strategy
            const potentialNegatives = indices.filter(idx => idx !== i && idx !== positiveIdx && 
                currentEmbeddings[idx].entityType !== anchor.entityType
            );

            if (potentialNegatives.length > 0) {
                if (params.miningStrategy === 'random') {
                    negativeIdx = potentialNegatives[Math.floor(Math.random() * potentialNegatives.length)];
                } else {
                    // Hard/Semi-hard Mining
                    // Find negatives that are CLOSE to anchor (distance < dist(A, P) + margin)
                    const distAP = positiveIdx !== -1 ? distSq(anchor.vector, currentEmbeddings[positiveIdx].vector) : 0;
                    
                    // Sort negatives by distance to anchor ASCENDING
                    const sortedNegatives = potentialNegatives
                        .map(idx => ({ idx, dist: distSq(anchor.vector, currentEmbeddings[idx].vector) }))
                        .sort((a, b) => a.dist - b.dist);

                    if (params.miningStrategy === 'hard') {
                         // Hardest negative: closest one
                        negativeIdx = sortedNegatives[0].idx;
                    } else {
                        // Semi-hard: Further than P, but within margin
                        const semiHard = sortedNegatives.find(n => n.dist > distAP && n.dist < distAP + params.margin);
                        negativeIdx = semiHard ? semiHard.idx : sortedNegatives[0].idx;
                    }
                }
            }

            if (positiveIdx === -1 || negativeIdx === -1) continue;

            const positive = currentEmbeddings[positiveIdx];
            const negative = currentEmbeddings[negativeIdx];

            // --- Loss Calculation ---
            const distPos = Math.sqrt(distSq(anchor.vector, positive.vector));
            const distNeg = Math.sqrt(distSq(anchor.vector, negative.vector));
            
            const loss = Math.max(distPos - distNeg + params.margin, 0);

            if (loss > 0) {
                totalLoss += loss;
                activeTriplets++;

                // --- Gradient Approximation ---
                // Gradients for Anchor, Positive, Negative
                // Gradient of dist w.r.t vector v is (v - target) / dist
                
                // Grad w.r.t Anchor: (A - P)/distP - (A - N)/distN
                const gradA_P = mult(sub(anchor.vector, positive.vector), 1 / (distPos + epsilon));
                const gradA_N = mult(sub(anchor.vector, negative.vector), 1 / (distNeg + epsilon));
                const gradA = sub(gradA_P, gradA_N); // Push to P, Pull from N

                // Grad w.r.t Positive: -(A - P)/distP (Pull to A)
                const gradP = mult(gradA_P, -1);

                // Grad w.r.t Negative: (A - N)/distN (Push away from A)
                const gradN = gradA_N;

                // --- AdamW Step ---
                applyAdamW(anchor.id, anchor.vector, gradA, learningRate, epoch, optimizerState, weightDecay, beta1, beta2, epsilon);
                applyAdamW(positive.id, positive.vector, gradP, learningRate, epoch, optimizerState, weightDecay, beta1, beta2, epsilon);
                applyAdamW(negative.id, negative.vector, gradN, learningRate, epoch, optimizerState, weightDecay, beta1, beta2, epsilon);
                
                // Re-normalize to keep on hypersphere (common in cosine/dot product spaces)
                anchor.vector = normalize(anchor.vector);
                positive.vector = normalize(positive.vector);
                negative.vector = normalize(negative.vector);
            }
        }

        // --- Learning Rate Scheduler (Linear Decay) ---
        learningRate = params.learningRate * (1 - (epoch / params.epochs));

        // Yield results to UI
        const avgLoss = activeTriplets > 0 ? totalLoss / activeTriplets : 0;
        
        // Simular delay para visualização
        await new Promise(r => setTimeout(r, 100));
        
        onEpochComplete({
            currentEpoch: epoch,
            loss: avgLoss,
            tripletCount: activeTriplets
        }, currentEmbeddings);
    }
};

// Helper: AdamW Update
function applyAdamW(
    id: string, 
    params: number[], 
    grads: number[], 
    lr: number, 
    t: number, 
    state: Record<string, { m: number[], v: number[] }>,
    weightDecay: number,
    beta1: number,
    beta2: number,
    epsilon: number
) {
    const s = state[id];
    
    for (let i = 0; i < params.length; i++) {
        // Weight Decay
        params[i] = params[i] * (1 - lr * weightDecay);

        // Update 1st moment
        s.m[i] = beta1 * s.m[i] + (1 - beta1) * grads[i];
        
        // Update 2nd moment
        s.v[i] = beta2 * s.v[i] + (1 - beta2) * (grads[i] * grads[i]);

        // Bias correction
        const mHat = s.m[i] / (1 - Math.pow(beta1, t));
        const vHat = s.v[i] / (1 - Math.pow(beta2, t));

        // Update parameter
        params[i] = params[i] - lr * (mHat / (Math.sqrt(vHat) + epsilon));
    }
}

function hasKeywordOverlap(a: EmbeddingVector, b: EmbeddingVector): boolean {
    if (!a.keywords || !b.keywords) return false;
    return a.keywords.some(k => b.keywords?.includes(k));
}
