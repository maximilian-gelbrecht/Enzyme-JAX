// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(enzyme-hlo-generate-td{patterns=concat_multipad},transform-interpreter,enzyme-hlo-remove-transform)" %s | FileCheck %s

// The merged pad keeps the high padding of the second pad along the
// concatenation axis.
func.func @two(%a: tensor<1x2x4xf32>, %b: tensor<1x5x4xf32>, %c: tensor<1x3x4xf32>, %pv: tensor<f32>) -> tensor<1x13x4xf32> {
  %pa = stablehlo.pad %a, %pv, low = [0, 1, 0], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<1x2x4xf32>, tensor<f32>) -> tensor<1x3x4xf32>
  %pb = stablehlo.pad %b, %pv, low = [0, 0, 0], high = [0, 2, 0], interior = [0, 0, 0] : (tensor<1x5x4xf32>, tensor<f32>) -> tensor<1x7x4xf32>
  %r = stablehlo.concatenate %pa, %pb, %c, dim = 1 : (tensor<1x3x4xf32>, tensor<1x7x4xf32>, tensor<1x3x4xf32>) -> tensor<1x13x4xf32>
  return %r : tensor<1x13x4xf32>
}

// CHECK:  func.func @two(%arg0: tensor<1x2x4xf32>, %arg1: tensor<1x5x4xf32>, %arg2: tensor<1x3x4xf32>, %arg3: tensor<f32>) -> tensor<1x13x4xf32> {
// CHECK-NEXT:    %0 = stablehlo.concatenate %arg0, %arg1, dim = 1 : (tensor<1x2x4xf32>, tensor<1x5x4xf32>) -> tensor<1x7x4xf32>
// CHECK-NEXT:    %1 = stablehlo.pad %0, %arg3, low = [0, 1, 0], high = [0, 2, 0], interior = [0, 0, 0] : (tensor<1x7x4xf32>, tensor<f32>) -> tensor<1x10x4xf32>
// CHECK-NEXT:    %2 = stablehlo.concatenate %1, %arg2, dim = 1 : (tensor<1x10x4xf32>, tensor<1x3x4xf32>) -> tensor<1x13x4xf32>
// CHECK-NEXT:    return %2 : tensor<1x13x4xf32>
// CHECK-NEXT:  }

// A chain of three pads merges into one pad with the high padding of the last.
func.func @three(%a: tensor<1x2x4xf32>, %b: tensor<1x5x4xf32>, %c: tensor<1x3x4xf32>, %d: tensor<2x1x4xf32>, %pv: tensor<f32>) -> tensor<2x15x4xf32> {
  %pa = stablehlo.pad %a, %pv, low = [1, 1, 0], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<1x2x4xf32>, tensor<f32>) -> tensor<2x3x4xf32>
  %pb = stablehlo.pad %b, %pv, low = [1, 0, 0], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<1x5x4xf32>, tensor<f32>) -> tensor<2x5x4xf32>
  %pc = stablehlo.pad %c, %pv, low = [1, 0, 0], high = [0, 3, 0], interior = [0, 0, 0] : (tensor<1x3x4xf32>, tensor<f32>) -> tensor<2x6x4xf32>
  %r = stablehlo.concatenate %pa, %pb, %pc, %d, dim = 1 : (tensor<2x3x4xf32>, tensor<2x5x4xf32>, tensor<2x6x4xf32>, tensor<2x1x4xf32>) -> tensor<2x15x4xf32>
  return %r : tensor<2x15x4xf32>
}

// CHECK:  func.func @three(%arg0: tensor<1x2x4xf32>, %arg1: tensor<1x5x4xf32>, %arg2: tensor<1x3x4xf32>, %arg3: tensor<2x1x4xf32>, %arg4: tensor<f32>) -> tensor<2x15x4xf32> {
// CHECK-NEXT:    %0 = stablehlo.concatenate %arg0, %arg1, dim = 1 : (tensor<1x2x4xf32>, tensor<1x5x4xf32>) -> tensor<1x7x4xf32>
// CHECK-NEXT:    %1 = stablehlo.concatenate %0, %arg2, dim = 1 : (tensor<1x7x4xf32>, tensor<1x3x4xf32>) -> tensor<1x10x4xf32>
// CHECK-NEXT:    %2 = stablehlo.pad %1, %arg4, low = [1, 1, 0], high = [0, 3, 0], interior = [0, 0, 0] : (tensor<1x10x4xf32>, tensor<f32>) -> tensor<2x14x4xf32>
// CHECK-NEXT:    %3 = stablehlo.concatenate %2, %arg3, dim = 1 : (tensor<2x14x4xf32>, tensor<2x1x4xf32>) -> tensor<2x15x4xf32>
// CHECK-NEXT:    return %3 : tensor<2x15x4xf32>
// CHECK-NEXT:  }

// Once the merged pad has high padding along the axis, no further pad is merged.
func.func @stop(%a: tensor<1x2x4xf32>, %b: tensor<1x5x4xf32>, %c: tensor<1x3x4xf32>, %pv: tensor<f32>) -> tensor<1x14x4xf32> {
  %pa = stablehlo.pad %a, %pv, low = [0, 1, 0], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<1x2x4xf32>, tensor<f32>) -> tensor<1x3x4xf32>
  %pb = stablehlo.pad %b, %pv, low = [0, 0, 0], high = [0, 2, 0], interior = [0, 0, 0] : (tensor<1x5x4xf32>, tensor<f32>) -> tensor<1x7x4xf32>
  %pc = stablehlo.pad %c, %pv, low = [0, 0, 0], high = [0, 1, 0], interior = [0, 0, 0] : (tensor<1x3x4xf32>, tensor<f32>) -> tensor<1x4x4xf32>
  %r = stablehlo.concatenate %pa, %pb, %pc, dim = 1 : (tensor<1x3x4xf32>, tensor<1x7x4xf32>, tensor<1x4x4xf32>) -> tensor<1x14x4xf32>
  return %r : tensor<1x14x4xf32>
}

// CHECK:  func.func @stop(%arg0: tensor<1x2x4xf32>, %arg1: tensor<1x5x4xf32>, %arg2: tensor<1x3x4xf32>, %arg3: tensor<f32>) -> tensor<1x14x4xf32> {
// CHECK-NEXT:    %0 = stablehlo.pad %arg2, %arg3, low = [0, 0, 0], high = [0, 1, 0], interior = [0, 0, 0] : (tensor<1x3x4xf32>, tensor<f32>) -> tensor<1x4x4xf32>
// CHECK-NEXT:    %1 = stablehlo.concatenate %arg0, %arg1, dim = 1 : (tensor<1x2x4xf32>, tensor<1x5x4xf32>) -> tensor<1x7x4xf32>
// CHECK-NEXT:    %2 = stablehlo.pad %1, %arg3, low = [0, 1, 0], high = [0, 2, 0], interior = [0, 0, 0] : (tensor<1x7x4xf32>, tensor<f32>) -> tensor<1x10x4xf32>
// CHECK-NEXT:    %3 = stablehlo.concatenate %2, %0, dim = 1 : (tensor<1x10x4xf32>, tensor<1x4x4xf32>) -> tensor<1x14x4xf32>
// CHECK-NEXT:    return %3 : tensor<1x14x4xf32>
// CHECK-NEXT:  }
