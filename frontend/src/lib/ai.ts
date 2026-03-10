export const NAMESPACES = [
  { id: 'customer_support', label: 'Customer Support', icon: '🎧' },
  { id: 'engineering', label: 'Engineering', icon: '🔧' },
  { id: 'dealer_sales', label: 'Dealer Sales', icon: '🚗' },
  { id: 'compliance', label: 'Compliance', icon: '📋' },
  { id: 'employee_hr', label: 'HR', icon: '👥' },
  { id: 'vendor', label: 'Vendor', icon: '📦' },
] as const;

export type Namespace = typeof NAMESPACES[number]['id'];
