# import tkinter as tk
# from tkinter import messagebox
# import random

# # 피자 종류와 가격 (조각당)
# pizza_prices = {
#     "페퍼로니": 3500,
#     "치즈": 3500,
#     "고구마": 4500,
#     "불고기": 4800
# }

# # 음료 종류와 가격
# drink_prices = {
#     "콜라": 1500,
#     "사이다": 1500,
#     "환타": 2000,
#     "생수": 1000
# }

# # 추가 토핑 가격
# topping_prices = {
#     "치즈 추가": 500,
#     "베이컨 추가": 1000,
#     "올리브 추가": 700
# }

# # 사이드 메뉴 가격
# side_prices = {
#     "감자튀김": 2500,
#     "치킨텐더": 3500,
#     "샐러드": 3000
# }

# def order_pizza():
#     total_price = 0
#     order_list = []

#     # 피자 주문 확인
#     for pizza, entry in slice_entries.items():
#         try:
#             slices = int(entry.get()) if entry.get() else 0
#             if slices < 0:
#                 raise ValueError
#         except ValueError:
#             messagebox.showerror("입력 오류", f"{pizza} 조각 수는 0 이상의 정수로 입력하세요.")
#             return

#         if slices > 0:
#             price = pizza_prices[pizza] * slices
#             order_list.append(f"{pizza} {slices}조각 ({price}원)")
#             total_price += price

#     # 음료 주문 확인
#     selected_drinks = []
#     for drink, var in drink_vars.items():
#         if var.get() == 1:
#             total_price += drink_prices[drink]
#             selected_drinks.append(f"{drink} ({drink_prices[drink]}원)")

#     # 토핑 주문 확인
#     selected_toppings = []
#     for topping, var in topping_vars.items():
#         if var.get() == 1:
#             total_price += topping_prices[topping]
#             selected_toppings.append(f"{topping} (+{topping_prices[topping]}원)")

#     # 사이드 주문 확인
#     selected_sides = []
#     for side, var in side_vars.items():
#         if var.get() == 1:
#             total_price += side_prices[side]
#             selected_sides.append(f"{side} ({side_prices[side]}원)")

#     # 주문 내역 확인
#     if not order_list and not selected_drinks and not selected_toppings and not selected_sides:
#         messagebox.showwarning("주문 없음", "최소 1조각 이상의 피자 또는 다른 메뉴를 선택하세요.")
#         return

#     # 결제 방식 & 배달 옵션
#     payment = payment_var.get()
#     order_type = order_type_var.get()

#     order_text = "주문 내역:\n"
#     if order_list:
#         order_text += "\n".join(order_list) + "\n"
#     if selected_drinks:
#         order_text += "음료: " + ", ".join(selected_drinks) + "\n"
#     if selected_toppings:
#         order_text += "토핑: " + ", ".join(selected_toppings) + "\n"
#     if selected_sides:
#         order_text += "사이드: " + ", ".join(selected_sides) + "\n"

#     order_text += f"\n결제 방식: {payment}\n수령 방법: {order_type}"
#     order_text += f"\n\n총 금액: {total_price}원"

#     result_label.config(text=order_text)

# def recommend_menu():
#     items = list(pizza_prices.keys()) + list(drink_prices.keys()) + list(side_prices.keys())
#     recommendation = random.choice(items)
#     messagebox.showinfo("오늘의 추천 메뉴", f"오늘의 추천 메뉴는 '{recommendation}' 입니다!")

# # 메인 윈도우
# root = tk.Tk()
# root.title("조각 피자 + 음료 주문 프로그램")
# root.geometry("420x500")

# # 스크롤 가능한 메인 프레임
# canvas = tk.Canvas(root)
# scrollbar = tk.Scrollbar(root, orient="vertical", command=canvas.yview)
# scroll_frame = tk.Frame(canvas)

# scroll_frame.bind(
#     "<Configure>",
#     lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
# )

# canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
# canvas.configure(yscrollcommand=scrollbar.set)

# canvas.pack(side="left", fill="both", expand=True)
# scrollbar.pack(side="right", fill="y")

# # ========== 주문 UI 영역 ==========
# # 피자 입력
# tk.Label(scroll_frame, text="피자 주문 (조각 수 입력)", font=("Arial", 12, "bold")).pack(pady=5)
# slice_entries = {}
# for pizza, price in pizza_prices.items():
#     frame = tk.Frame(scroll_frame)
#     frame.pack(anchor="w", padx=10, pady=2)
#     tk.Label(frame, text=f"{pizza} ({price}원/조각)").pack(side="left")
#     entry = tk.Entry(frame, width=5)
#     entry.pack(side="left", padx=5)
#     slice_entries[pizza] = entry

# # 음료 선택
# tk.Label(scroll_frame, text="음료 선택", font=("Arial", 12, "bold")).pack(pady=10)
# drink_vars = {}
# for drink, price in drink_prices.items():
#     var = tk.IntVar()
#     chk = tk.Checkbutton(scroll_frame, text=f"{drink} ({price}원)", variable=var)
#     chk.pack(anchor="w", padx=20)
#     drink_vars[drink] = var

# # 토핑 선택
# tk.Label(scroll_frame, text="추가 토핑 선택", font=("Arial", 12, "bold")).pack(pady=10)
# topping_vars = {}
# for topping, price in topping_prices.items():
#     var = tk.IntVar()
#     chk = tk.Checkbutton(scroll_frame, text=f"{topping} (+{price}원)", variable=var)
#     chk.pack(anchor="w", padx=20)
#     topping_vars[topping] = var

# # 사이드 메뉴 선택
# tk.Label(scroll_frame, text="사이드 메뉴 선택", font=("Arial", 12, "bold")).pack(pady=10)
# side_vars = {}
# for side, price in side_prices.items():
#     var = tk.IntVar()
#     chk = tk.Checkbutton(scroll_frame, text=f"{side} ({price}원)", variable=var)
#     chk.pack(anchor="w", padx=20)
#     side_vars[side] = var

# # 결제 방식 선택
# tk.Label(scroll_frame, text="결제 방식 선택", font=("Arial", 12, "bold")).pack(pady=10)
# payment_var = tk.StringVar(value="현금")
# for method in ["현금", "카드", "간편결제"]:
#     tk.Radiobutton(scroll_frame, text=method, variable=payment_var, value=method).pack(anchor="w", padx=20)

# # 수령 방법 선택
# tk.Label(scroll_frame, text="수령 방법 선택", font=("Arial", 12, "bold")).pack(pady=10)
# order_type_var = tk.StringVar(value="매장식사")
# for otype in ["매장식사", "포장", "배달"]:
#     tk.Radiobutton(scroll_frame, text=otype, variable=order_type_var, value=otype).pack(anchor="w", padx=20)

# # ========== 버튼 & 결과 출력 ==========
# button_frame = tk.Frame(root)
# button_frame.pack(fill="x", pady=5)

# order_button = tk.Button(button_frame, text="주문하기", command=order_pizza, bg="orange", fg="white", font=("Arial", 12))
# order_button.pack(side="left", expand=True, fill="x", padx=5)

# recommend_button = tk.Button(button_frame, text="오늘의 추천 메뉴", command=recommend_menu, bg="green", fg="white", font=("Arial", 11))
# recommend_button.pack(side="right", expand=True, fill="x", padx=5)

# result_label = tk.Label(root, text="", font=("Arial", 11), fg="blue", justify="left")
# result_label.pack(pady=10)

# root.mainloop()
